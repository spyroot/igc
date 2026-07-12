"""M3 goal planner: NLP instruction to structured goal plan.

M3 is the goal planner / goal extractor stage. It converts an operator sentence
plus context into an auditable, machine-checkable plan that M4 can verify and
M6 can condition on. The output is structured JSON, not a latent vector.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from igc.core.types import Goal

M3_SCHEMA_VERSION = "m3.goal_plan.v1"
M3_STAGE_NAME = "M3_GOAL_PLANNER"
M3_PROPER_NAME = "Goal Planner / Goal Extractor"


@dataclass
class M3GoalPlannerContext:
    """Context used to map text to a safe goal plan.

    :param state_summary: Compact observed state, such as installed OS targets,
        boot override state, BIOS attributes, or virtual-media status.
    :param tool_catalog: Legal tool/action declarations visible to the planner.
    :param approved_images: Approved install/boot image records. M3 may infer an
        ISO mount only from this catalog, never by inventing a URL.
    :param safety_policy: Local safety constraints and approval policy.
    """

    state_summary: dict[str, Any] = field(default_factory=dict)
    tool_catalog: list[dict[str, Any]] = field(default_factory=list)
    approved_images: list[dict[str, Any]] = field(default_factory=list)
    safety_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Optional[dict[str, Any]]) -> "M3GoalPlannerContext":
        """Build context from a plain mapping.

        :param value: Optional mapping with ``state_summary``, ``tool_catalog``,
            ``approved_images``, and ``safety_policy`` keys.
        :return: Context instance with missing fields defaulted.
        """
        value = value or {}
        return cls(
            state_summary=dict(value.get("state_summary") or {}),
            tool_catalog=list(value.get("tool_catalog") or []),
            approved_images=list(value.get("approved_images") or []),
            safety_policy=dict(value.get("safety_policy") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class M3DerivedRequirement:
    """Requirement inferred by M3 from instruction plus context."""

    name: str
    reason: str
    requires: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


@dataclass
class M3SubGoal:
    """One machine-checkable sub-goal in the M3 plan."""

    subgoal_id: str
    op: str
    success_predicate: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    risk_level: str = "READ_ONLY"


@dataclass
class M3GoalPlan:
    """Structured M3 output consumed by M4 and M6.

    :param goal_id: Stable goal identifier derived from the instruction.
    :param instruction: Original operator instruction.
    :param final_spec: Machine-checkable final state spec.
    :param constraints: Safety and execution constraints.
    :param derived_requirements: Requirements inferred from context.
    :param plan: Ordered sub-goals. Each sub-goal has a verifier predicate.
    :param blocked_reasons: Blocking reasons, if M3 cannot safely fill a field.
    :param context_refs: Optional provenance names for catalogs/context used.
    """

    goal_id: str
    instruction: str
    final_spec: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    derived_requirements: list[M3DerivedRequirement] = field(default_factory=list)
    plan: list[M3SubGoal] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    context_refs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = M3_SCHEMA_VERSION
    stage: str = M3_STAGE_NAME
    proper_name: str = M3_PROPER_NAME

    @property
    def blocked(self) -> bool:
        """Whether this plan needs human or catalog input before execution."""
        return bool(self.blocked_reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical public JSON shape for M3 output."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "proper_name": self.proper_name,
            "goal_id": self.goal_id,
            "instruction": self.instruction,
            "final_spec": self.final_spec,
            "constraints": self.constraints,
            "derived_requirements": [asdict(item) for item in self.derived_requirements],
            "plan": [asdict(item) for item in self.plan],
            "blocked": self.blocked,
            "blocked_reasons": self.blocked_reasons,
            "context_refs": self.context_refs,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "M3GoalPlan":
        """Reconstruct a plan from model or dataset JSON."""
        return cls(
            schema_version=value.get("schema_version", M3_SCHEMA_VERSION),
            stage=value.get("stage", M3_STAGE_NAME),
            proper_name=value.get("proper_name", M3_PROPER_NAME),
            goal_id=value["goal_id"],
            instruction=value["instruction"],
            final_spec=dict(value.get("final_spec") or {}),
            constraints=list(value.get("constraints") or []),
            derived_requirements=[
                M3DerivedRequirement(**item)
                for item in value.get("derived_requirements", [])
            ],
            plan=[M3SubGoal(**item) for item in value.get("plan", [])],
            blocked_reasons=list(value.get("blocked_reasons") or []),
            context_refs=dict(value.get("context_refs") or {}),
        )

    def to_goal(self) -> Goal:
        """Convert to the repo-wide ``Goal`` contract."""
        return Goal(
            instruction=self.instruction,
            spec=dict(self.final_spec),
            plan=[asdict(item) for item in self.plan],
            constraints=list(self.constraints) + list(self.blocked_reasons),
        )


class M3GoalPlanJsonCodec:
    """Prompt and JSON codec for M3 SFT and inference."""

    input_sentinel = "<|m3_goal_planner_input|>"
    output_sentinel = "<|m3_goal_planner_output|>"

    @staticmethod
    def dumps(plan: M3GoalPlan) -> str:
        """Serialize a plan as stable JSON for training targets."""
        return json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def loads(text: str) -> M3GoalPlan:
        """Parse the first JSON object in model output into an M3 plan."""
        raw = _extract_first_json_object(text)
        return M3GoalPlan.from_dict(json.loads(raw))

    @classmethod
    def build_prompt(cls, instruction: str, context: M3GoalPlannerContext | dict | None) -> str:
        """Build the supervised prompt used by M3 training and inference."""
        planner_context = (
            context if isinstance(context, M3GoalPlannerContext)
            else M3GoalPlannerContext.from_mapping(context)
        )
        payload = {
            "task": "Convert the operator instruction into an M3 goal plan JSON object.",
            "schema_version": M3_SCHEMA_VERSION,
            "output_contract": {
                "final_spec": "machine-checkable desired state",
                "constraints": "safety constraints and approval requirements",
                "derived_requirements": "safe requirements inferred from context",
                "plan": "ordered subgoals with success_predicate fields",
            },
            "instruction": instruction,
            "context": planner_context.to_dict(),
        }
        return (
            f"{cls.input_sentinel}\n"
            f"{json.dumps(payload, sort_keys=True)}\n"
            f"{cls.output_sentinel}\n"
        )


class M3DeterministicGoalPlanner:
    """Rule-backed M3 planner for boot, ISO, BIOS, and safe fallback cases.

    This is both an immediate usable planner and a data generator for SFT. A
    trained model should match this output shape, not hide decisions in latent
    state.
    """

    def plan(
        self,
        instruction: str,
        context: M3GoalPlannerContext | dict | None = None,
    ) -> M3GoalPlan:
        """Convert an operator sentence into a structured M3 goal plan."""
        planner_context = (
            context if isinstance(context, M3GoalPlannerContext)
            else M3GoalPlannerContext.from_mapping(context)
        )
        text = _normalize_text(instruction)
        if _mentions_ubuntu_2204(text) or _mentions_fast_boot(text):
            return self._plan_ubuntu_fast_boot(instruction, planner_context)
        action_record = _match_catalog_action(text, planner_context.tool_catalog)
        if action_record is not None:
            return build_redfish_action_goal_plan(instruction, action_record, planner_context)
        return self._plan_unknown(instruction, planner_context)

    def _plan_ubuntu_fast_boot(
        self,
        instruction: str,
        context: M3GoalPlannerContext,
    ) -> M3GoalPlan:
        """Build the boot Ubuntu 22.04 + fast-boot plan."""
        final_spec = {}
        constraints = [
            "approval_required_for_mutation",
            "no_guessed_image_uri",
            "verify_after_reboot",
            "idempotent",
        ]
        derived: list[M3DerivedRequirement] = []
        subgoals: list[M3SubGoal] = [
            M3SubGoal(
                subgoal_id="discover_boot_state",
                op="discover_boot_state",
                success_predicate="boot_state_known",
            )
        ]
        blocked_reasons: list[str] = []

        if _mentions_ubuntu_2204(_normalize_text(instruction)):
            final_spec["os.version"] = "ubuntu-22.04"
            installed = _has_verified_installed_os(context, "ubuntu-22.04")
            image = _find_approved_image(context, os_name="ubuntu", version="22.04")
            if installed:
                derived.append(M3DerivedRequirement(
                    name="use_installed_boot_target",
                    reason="verified ubuntu-22.04 boot target already exists",
                    evidence={"os": "ubuntu-22.04"},
                ))
                subgoals.append(M3SubGoal(
                    subgoal_id="set_boot_override_installed_os_once",
                    op="set_boot_override",
                    success_predicate="boot.override == installed_ubuntu_22_04_once",
                    arguments={"target": "installed_ubuntu_22_04", "mode": "Once"},
                    risk_level="MUTATING",
                ))
            elif image is not None:
                image_ref = image.get("image_id") or image.get("name") or image.get("uri")
                derived.append(M3DerivedRequirement(
                    name="mount_install_media",
                    reason="ubuntu-22.04 requested and no verified installed boot target is known",
                    requires=["approved_iso_ref"],
                    evidence={"approved_iso_ref": image_ref},
                ))
                subgoals.extend([
                    M3SubGoal(
                        subgoal_id="discover_virtual_media",
                        op="discover_virtual_media",
                        success_predicate="virtual_media_slots_known",
                    ),
                    M3SubGoal(
                        subgoal_id="mount_ubuntu_2204_iso",
                        op="mount_virtual_media",
                        success_predicate=(
                            "virtual_media.inserted == true && "
                            "virtual_media.image_ref == approved_ubuntu_22_04_iso"
                        ),
                        arguments={"image_ref": image_ref, "write_protected": True},
                        requires=["discover_virtual_media"],
                        risk_level="MUTATING",
                    ),
                    M3SubGoal(
                        subgoal_id="set_boot_override_virtual_cd_once",
                        op="set_boot_override",
                        success_predicate="boot.override == virtual_cd_once",
                        arguments={"target": "virtual_cd", "mode": "Once"},
                        requires=["mount_ubuntu_2204_iso"],
                        risk_level="MUTATING",
                    ),
                ])
            else:
                derived.append(M3DerivedRequirement(
                    name="mount_install_media",
                    reason="ubuntu-22.04 requested and no verified installed boot target is known",
                    requires=["approved_iso_ref"],
                    blocked=True,
                ))
                blocked_reasons.append("BLOCKED_NEEDS_APPROVED_UBUNTU_22_04_ISO")

        if _mentions_fast_boot(_normalize_text(instruction)):
            final_spec["bios.fast_boot"] = True
            subgoals.extend([
                M3SubGoal(
                    subgoal_id="discover_bios_settings",
                    op="discover_bios_settings",
                    success_predicate="bios_settings_object_known",
                ),
                M3SubGoal(
                    subgoal_id="set_bios_fast_boot",
                    op="set_bios_attribute",
                    success_predicate="pending_or_current_bios.fast_boot == true",
                    arguments={"attribute": "fast_boot", "value": True},
                    requires=["discover_bios_settings"],
                    risk_level="MUTATING",
                ),
            ])

        subgoals.extend([
            M3SubGoal(
                subgoal_id="apply_pending_settings",
                op="apply_pending_settings",
                success_predicate="pending_settings_empty == true",
                risk_level="MUTATING",
            ),
            M3SubGoal(
                subgoal_id="reboot_if_required",
                op="reboot_if_required",
                success_predicate="reboot_completed_or_not_required == true",
                requires=["apply_pending_settings"],
                risk_level="DESTRUCTIVE",
            ),
            M3SubGoal(
                subgoal_id="verify_final_goal",
                op="verify_goal",
                success_predicate="final_spec_satisfied == true",
                requires=["reboot_if_required"],
            ),
        ])

        return M3GoalPlan(
            goal_id=_goal_id(instruction),
            instruction=instruction,
            final_spec=final_spec,
            constraints=constraints,
            derived_requirements=derived,
            plan=subgoals,
            blocked_reasons=blocked_reasons,
            context_refs={
                "approved_image_count": len(context.approved_images),
                "tool_count": len(context.tool_catalog),
            },
        )

    def _plan_unknown(self, instruction: str, context: M3GoalPlannerContext) -> M3GoalPlan:
        """Return a safe blocked plan for unsupported instructions."""
        return M3GoalPlan(
            goal_id=_goal_id(instruction),
            instruction=instruction,
            final_spec={"instruction_text": instruction},
            constraints=["approval_required_for_mutation", "no_unverified_execution"],
            derived_requirements=[],
            plan=[
                M3SubGoal(
                    subgoal_id="clarify_or_distill_goal",
                    op="clarify_goal",
                    success_predicate="machine_checkable_goal_spec_available == true",
                )
            ],
            blocked_reasons=["BLOCKED_UNSUPPORTED_GOAL_TEMPLATE"],
            context_refs={
                "approved_image_count": len(context.approved_images),
                "tool_count": len(context.tool_catalog),
            },
        )


class M3ModelGoalPlanner:
    """M3 planner backed by a causal language model checkpoint."""

    def __init__(self, model, tokenizer, fallback: Optional[M3DeterministicGoalPlanner] = None):
        """Create a model-backed planner.

        :param model: HuggingFace causal LM with ``generate``.
        :param tokenizer: Matching tokenizer.
        :param fallback: Optional deterministic planner used when JSON parsing
            fails.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.fallback = fallback

    def plan(
        self,
        instruction: str,
        context: M3GoalPlannerContext | dict | None = None,
        *,
        max_new_tokens: int = 1024,
    ) -> M3GoalPlan:
        """Generate a structured M3 plan from a trained checkpoint."""
        planner_context = (
            context if isinstance(context, M3GoalPlannerContext)
            else M3GoalPlannerContext.from_mapping(context)
        )
        prompt = M3GoalPlanJsonCodec.build_prompt(instruction, planner_context)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = encoded["input_ids"].shape[-1]
        generated = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        try:
            return M3GoalPlanJsonCodec.loads(generated)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            if self.fallback is None:
                raise
            return self.fallback.plan(instruction, planner_context)


def build_redfish_action_goal_plan(
    instruction: str,
    action_record: dict[str, Any],
    context: M3GoalPlannerContext | None = None,
) -> M3GoalPlan:
    """Build an M3 plan for one discovered Redfish action.

    :param instruction: Operator instruction used as the supervised prompt.
    :param action_record: Action declaration with keys such as ``action``,
        ``target``, ``method``, ``allowed_methods``, and ``arguments``.
    :param context: Optional planner context for provenance counters.
    :return: Structured plan that discovers, validates, executes, and verifies
        the Redfish action.
    """
    context = context or M3GoalPlannerContext()
    action_name = _action_name(action_record)
    target = _action_target(action_record)
    method = _action_method(action_record)
    risk_level = _risk_for_method_and_action(method, action_name)
    arguments = dict(
        action_record.get("arguments")
        or action_record.get("argument_schema")
        or action_record.get("parameters")
        or {}
    )
    constraints = [
        "legal_action_required",
        "verify_after_action",
        "idempotent_when_current_state_satisfies_spec",
    ]
    if risk_level != "READ_ONLY":
        constraints.append("approval_required_for_mutation")
    final_spec = {
        "redfish.action": action_name,
        "redfish.target": target,
        "redfish.method": method,
        "redfish.effect_verified": True,
    }
    if arguments:
        final_spec["redfish.arguments"] = arguments

    plan = [
        M3SubGoal(
            subgoal_id=f"discover_{_safe_id(action_name)}_resource",
            op="discover_redfish_resource",
            success_predicate=f"resource_exists('{target}') == true",
            arguments={"target": target},
        ),
        M3SubGoal(
            subgoal_id=f"validate_{_safe_id(action_name)}_legal_action",
            op="validate_legal_action",
            success_predicate=(
                f"method_allowed('{target}', '{method}') == true && "
                f"action_available('{action_name}') == true"
            ),
            arguments={"target": target, "method": method, "action": action_name},
            requires=[f"discover_{_safe_id(action_name)}_resource"],
        ),
        M3SubGoal(
            subgoal_id=f"execute_{_safe_id(action_name)}",
            op="execute_redfish_action",
            success_predicate=f"redfish_response_success('{action_name}') == true",
            arguments={
                "action": action_name,
                "target": target,
                "method": method,
                "body": arguments,
            },
            requires=[f"validate_{_safe_id(action_name)}_legal_action"],
            risk_level=risk_level,
        ),
        M3SubGoal(
            subgoal_id=f"verify_{_safe_id(action_name)}_effect",
            op="verify_redfish_action_effect",
            success_predicate=f"redfish_effect_verified('{action_name}') == true",
            arguments={"action": action_name, "target": target},
            requires=[f"execute_{_safe_id(action_name)}"],
        ),
    ]
    return M3GoalPlan(
        goal_id=_goal_id(instruction),
        instruction=instruction,
        final_spec=final_spec,
        constraints=constraints,
        derived_requirements=[
            M3DerivedRequirement(
                name="redfish_action_catalog_entry",
                reason="instruction maps to a discovered Redfish action",
                evidence={
                    "action": action_name,
                    "target": target,
                    "method": method,
                },
            )
        ],
        plan=plan,
        context_refs={
            "approved_image_count": len(context.approved_images),
            "tool_count": len(context.tool_catalog),
        },
    )


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from generated text."""
    start = text.find("{")
    if start < 0:
        raise ValueError("model output did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    raise ValueError("model output contained an incomplete JSON object")


def _normalize_text(value: str) -> str:
    """Normalize instruction text for deterministic planning rules."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _mentions_ubuntu_2204(text: str) -> bool:
    """Whether the instruction asks for Ubuntu 22.04."""
    return bool(re.search(r"ubuntu\s*22[.\s-]?04|ubuntu2204", text))


def _mentions_fast_boot(text: str) -> bool:
    """Whether the instruction asks for fast boot BIOS behavior."""
    return "fast boot" in text or "fastboot" in text


def _goal_id(instruction: str) -> str:
    """Create a stable readable goal id from an instruction."""
    stem = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")
    return stem[:80] or "m3_goal"


def _has_verified_installed_os(context: M3GoalPlannerContext, os_version: str) -> bool:
    """Check whether the observed context already has a verified installed OS."""
    state = context.state_summary
    candidates = []
    for key in ("installed_os", "verified_installed_os", "boot_targets"):
        value = state.get(key)
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(str(item) for item in value)
        elif isinstance(value, dict):
            candidates.extend(str(item) for item in value.values())
    target = os_version.lower().replace("-", "").replace(".", "")
    return any(target in item.lower().replace("-", "").replace(".", "") for item in candidates)


def _find_approved_image(
    context: M3GoalPlannerContext,
    *,
    os_name: str,
    version: str,
) -> Optional[dict[str, Any]]:
    """Find an approved image matching an OS/version request."""
    version_key = version.lower().replace(".", "")
    for image in context.approved_images:
        if image.get("approved") is False:
            continue
        haystack = " ".join(str(value) for value in image.values()).lower()
        if os_name.lower() in haystack and version_key in haystack.replace(".", ""):
            return image
    return None


def _match_catalog_action(
    normalized_instruction: str,
    tool_catalog: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find the best Redfish action record mentioned by an instruction."""
    best: tuple[int, dict[str, Any]] | None = None
    instruction_tokens = set(_tokens(normalized_instruction))
    for record in tool_catalog:
        action_name = _action_name(record)
        target = _action_target(record)
        candidates = [
            _normalize_text(action_name),
            _normalize_text(action_name.replace(".", " ")),
            _normalize_text(str(record.get("op", ""))),
            _normalize_text(target),
        ]
        score = 0
        for candidate in candidates:
            if candidate and candidate in normalized_instruction:
                score += 8
        action_tokens = set(_tokens(action_name))
        if action_tokens:
            score += len(instruction_tokens & action_tokens)
        if score > 0 and (best is None or score > best[0]):
            best = (score, record)
    return best[1] if best else None


def _action_name(action_record: dict[str, Any]) -> str:
    """Return a stable action name from a Redfish action record."""
    return str(
        action_record.get("action")
        or action_record.get("name")
        or action_record.get("op")
        or action_record.get("operation")
        or "redfish_action"
    )


def _action_target(action_record: dict[str, Any]) -> str:
    """Return target URI/endpoint from a Redfish action record."""
    return str(
        action_record.get("target")
        or action_record.get("uri")
        or action_record.get("endpoint")
        or action_record.get("rest_api")
        or ""
    )


def _action_method(action_record: dict[str, Any]) -> str:
    """Infer the HTTP method for a Redfish action record."""
    explicit = action_record.get("method")
    if explicit:
        return str(explicit).upper()
    allowed = [str(item).upper() for item in action_record.get("allowed_methods", [])]
    for method in ("POST", "PATCH", "PUT", "DELETE", "GET"):
        if method in allowed:
            return method
    action_name = _action_name(action_record).lower()
    target = _action_target(action_record).lower()
    if "actions/" in target or "reset" in action_name or "insertmedia" in action_name:
        return "POST"
    if "delete" in action_name or "remove" in action_name or "eject" in action_name:
        return "POST"
    if "set" in action_name or "update" in action_name or "patch" in action_name:
        return "PATCH"
    return "GET"


def _risk_for_method_and_action(method: str, action_name: str) -> str:
    """Assign conservative risk from method/action text."""
    name = action_name.lower()
    if method == "GET":
        return "READ_ONLY"
    if any(word in name for word in ("reset", "reboot", "power", "force", "delete", "erase")):
        return "DESTRUCTIVE"
    return "MUTATING"


def _safe_id(value: str) -> str:
    """Return a readable identifier fragment."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "redfish_action"


def _tokens(value: str) -> list[str]:
    """Tokenize action names for catalog matching."""
    return [
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if len(token) > 2 and token not in {"redfish", "actions", "action"}
    ]


# Author: Mus mbayramo@stanford.edu
