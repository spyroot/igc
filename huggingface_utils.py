from transformers import TrainerCallback


class LossMonitorCallback(TrainerCallback):
    def __init__(self, logging_steps=10):
        self.logging_steps = logging_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.logging_steps == 0:
            print(f"Step: {state.global_step}, Loss: {logs['loss']}")
