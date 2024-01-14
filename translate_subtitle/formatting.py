def to_str_keys(input_: dict[int, str]) -> dict[str, str]:
    return {str(k): v for k, v in input_.items()}


def to_int_keys(input_: dict[str, str]) -> dict[int, str]:
    return {int(k): v for k, v in input_.items()}
