"""Keep JSON consumers on the last accepted nonempty value during OCR gaps."""

from text_detection_target import TextDetectionTargetWithResult


class JsonOutput:
    def __init__(self):
        self.last_values = {}

    def update(self, results):
        states = TextDetectionTargetWithResult.ResultState
        current_names = {result.name for result in results}
        self.last_values = {
            name: value
            for name, value in self.last_values.items()
            if name in current_names
        }
        snapshot, pivot = [], {}
        for result in results:
            accepted = result.result_state in (states.Success, states.SameNoChange)
            if accepted and result.result:
                self.last_values[result.name] = result.result
            elif result.name in self.last_values:
                # Copy only the output view; retain the actual detection state
                # and leave the OCR result used by other integrations untouched.
                result = TextDetectionTargetWithResult(
                    result,
                    self.last_values[result.name],
                    result.result_state,
                    result.effectiveRect,
                    result.extras,
                )
                pivot[result.name] = result.result
            snapshot.append(result)
            if accepted or (
                result.result_state == states.Empty
                and not (result.settings or {}).get("skip_empty", False)
            ):
                pivot[result.name] = result.result
        return snapshot, pivot
