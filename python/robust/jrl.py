import gtsam
import jrl

# ------------------------- Import according cpp helpers as well ------------------------- #
from robust.robust_python import makeRobustParser, makeRobustWriter


# ------------------------- Helpers for Wrapping GTSAM containers ------------------------- #
def values2results(
    values: gtsam.Values, name: str = "", dataset: str = ""
) -> jrl.Results:
    type_values = values2typedvalues(values)
    return jrl.Results(dataset, name, ["a"], {"a": type_values})


def values2typedvalues(values: gtsam.Values) -> jrl.TypedValues:
    types = {}
    for k in values.keys():
        char = chr(gtsam.Symbol(k).chr())
        if char in ["x", "w"]:
            types[k] = jrl.Pose3Tag
        elif char in ["l", "v", "i"]:
            types[k] = jrl.Point3Tag
        elif char == "b":
            types[k] = jrl.IMUBiasTag
        elif char == "s":
            types[k] = jrl.Point2Tag
        else:
            raise ValueError(f"Unknown character {char} in values")

    return jrl.TypedValues(values, types)
