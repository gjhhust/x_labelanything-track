#!/bin/bash
python_file="anylabeling/app_info.py"
variable_name="__preferred_device__"
variable_value=$(python3 - <<END
import re
with open("anylabeling/app_info.py", "r") as f:
    content = f.read()
target_variable = "__preferred_device__"
pattern = r'{} = "(.*?)"'.format(target_variable)
match = re.search(pattern, content)
if match:
    print(match.group(1))
END
)
echo "$variable_name: $variable_value"
system=$1
if [ "$system" = "win-cpu" ]; then
    expected_value="CPU"
    if [ "$variable_value" = "$expected_value" ]; then
        pyinstaller --noconfirm anylabeling-win-cpu.spec
    else
        echo "Variable $variable_name has a different value: $variable_value (expected: $expected_value)"
    fi
else
    echo "System value '$system' is not recognized."
fi