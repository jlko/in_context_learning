
def autotype(x):
    """Auto-converts string to bool/int/float if possible."""
    assert isinstance(x, str)
    if x.lower() in {'true', 'false'}:
        return x.lower() == 'true'  # Returns as bool.
    try:
        return int(x)  # Returns as int.
    except ValueError:
        try:
            return float(x)  # Returns as float.
        except ValueError:
            return x  # Returns as str.


def get_nested_dict(name, value):
    update_dict = {}
    update = update_dict
    splits = name.split('.')
    for i, n in enumerate(splits):

        if i != len(splits) - 1:
            update[n] = {}
        else:
            value = autotype(value)

            if isinstance(value, str) and ',' in value:
                # Detect lists:
                # remove brackets
                value = value[1:-1].split(',')
                # remove quotation marks, whitespaces
                value = [
                    x.strip().replace('\'', '').replace('"', '')
                    for x in value]
                value = [autotype(v) for v in value]
                if value == 'none':
                    value = None

            update[n] = value
        update = update[n]
    return update_dict
