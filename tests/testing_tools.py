def get_difference_stats(diff):
    """

    :param diff:
    :return:
    """
    added_lines = 0
    removed_lines = 0
    modified_lines = 0

    for line in diff:
        if line.startswith('+'):
            added_lines += 1
        elif line.startswith('-'):
            removed_lines += 1
        elif line.startswith('?'):
            modified_lines += 1

    return added_lines, removed_lines, modified_lines
