import json
import os
import sys
from enum import Enum

from igc.modules.igc_main import IgcMain
from igc.shared.shared_main import shared_main
from igc.shared.shared_torch_builder import TorchBuilder


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


# Argument keys whose values are credentials/secrets (e.g. --redfish-password, --x-auth).
# They must never be serialized into the saved parameters.json, which travels with the
# checkpoint (publish_checkpoint.sh) and the experiment dir.
_SENSITIVE_KEY_PARTS = ("password", "token", "secret", "api_key", "apikey", "x_auth")


def _is_sensitive(key: str) -> bool:
    """True if an argument key names a credential whose value must be redacted."""
    k = key.lower()
    return any(part in k for part in _SENSITIVE_KEY_PARTS)


def save_spec(cmd, parser_groups):
    """
    Save the parameters as a JSON file.

    :param cmd: The argparse.Namespace object containing the program arguments.
    :param parser_groups: List of tuples (section_name, section_parser).
    """
    if not os.path.isdir(cmd.output_dir):
        print(f"Error: Invalid output directory: {cmd.output_dir}. "
              "Please specify a valid directory.", file=sys.stderr)
        sys.exit(1)

    params = {}
    for section_name, section_parser in parser_groups:
        section_args, _ = section_parser.parse_known_args()
        section_keys = vars(section_args).keys()
        params[section_name] = {
            key: ("***REDACTED***" if _is_sensitive(key) and getattr(section_args, key)
                  else getattr(section_args, key))
            for key in section_keys
        }
        if 'device' not in params[section_name]:
            params[section_name]['device'] = None

        params[section_name]['device'] = str(params[section_name]['device'])

    save_path = os.path.join(cmd.output_dir, "parameters.json")
    print(f"Saved model trainer in {save_path}")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4, cls=EnumEncoder)


def main(cmd, parser_groups):
    """
    :return:
    """
    # Enable TF32 tensor-core matmuls (free speedup on Ampere+/Blackwell) when --tf32.
    TorchBuilder.enable_perf_backends(getattr(cmd, "tf32", False))
    save_spec(cmd, parser_groups)
    igc = IgcMain(cmd)
    igc.run()


if __name__ == '__main__':
    args, groups = shared_main()
    print(f"Starting args.local_rank {args.local_rank}, device: {args.device}")
    main(args, groups)
