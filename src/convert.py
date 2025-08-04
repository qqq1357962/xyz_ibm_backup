import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_parent_dir, "data_openroad/superblue2/superblue2.def")
output_path = os.path.join(parent_parent_dir, "data_openroad/superblue2/2_1_floorplan.odb.def")

read_components = False
has_read_components = False
with open(file_path, "r") as f:
    with open(output_path, "w") as out_f:
        for line in f:
            if not has_read_components:
                string = ""
                string += line[:len(line) - len(line.lstrip())]
                for cell in line.split():
                    if cell == "COMPONENTS":
                        read_components = True
                    if read_components and cell == "END":
                        read_components = False
                        has_read_components = True

                    if read_components:
                        if cell.startswith("+"):
                            string += ";"
                            break
                        elif cell != line.split()[-1]:
                            string += cell + " "
                        else:
                            string += cell
                    elif cell != line.split()[-1]:
                        string += cell + " "
                    else:
                        string += cell

                string += "\n"
                out_f.write(string)
            else:
                out_f.write(line)