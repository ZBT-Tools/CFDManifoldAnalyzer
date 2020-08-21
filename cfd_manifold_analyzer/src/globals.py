import re

SMALL = 1e-8


# replacing substring tools
def find_number(main_str, sub_str):
    return re.findall(r'%s(\d+)' % sub_str, main_str)


def replace_number(main_str, sub_str, number):
    old_number = find_number(main_str, sub_str)[0]
    return main_str.replace(sub_str + str(old_number), sub_str + str(number))
