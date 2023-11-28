import os

bar = "│"
elbow = "└──"
inter = "├──"
tab = "  "

ignore = ["__pycache__"]

def create_tree(root, depth=1, buffer=""):
    if (depth == 1):
        print(root)
    sub_contents = os.listdir(root)
    cont_buffer = buffer
    buffer = buffer + bar + tab
    for item in ignore:
        if item in sub_contents:
            sub_contents.remove(item)
    num_dir = len(sub_contents)
    
    if num_dir != 0:
        print(buffer)

    for i, content in enumerate(sub_contents):
        full_cont_path = os.path.join(root, content)
        if i != num_dir - 1:
            print(cont_buffer + inter + content, end="")
        else:
            print(cont_buffer + elbow + content, end="")
        
        if os.path.isdir(full_cont_path):
            print("/")
            next_buff = cont_buffer
            if i == num_dir - 1:
                next_buff += " " + tab
            else:
                next_buff = buffer
            create_tree(full_cont_path, depth + 1, next_buff)
        else:
            print()
        
        if i != num_dir - 1:
            print(buffer)

def create_tree_file(root, out_file, depth=1, buffer=""):
    if (depth == 1):
        out_file.write(root + "\n")
    sub_contents = os.listdir(root)
    cont_buffer = buffer
    buffer = buffer + bar + tab
    for item in ignore:
        if item in sub_contents:
            sub_contents.remove(item)
    num_dir = len(sub_contents)
    
    if num_dir != 0:
        out_file.write(buffer + "\n")
        
    for i, content in enumerate(sub_contents):
        full_cont_path = os.path.join(root, content)
        if i != num_dir - 1:
            out_file.write(cont_buffer + inter + content)
        else:
            out_file.write(cont_buffer + elbow + content)
        
        if os.path.isdir(full_cont_path):
            out_file.write("/" + "\n")
            next_buff = cont_buffer
            if i == num_dir - 1:
                next_buff += " " + tab
            else:
                next_buff = buffer
            create_tree_file(full_cont_path, out_file, depth + 1, next_buff)
        else:
            out_file.write("\n")
        
        if i != num_dir - 1:
            out_file.write(buffer + "\n")

def main():
    root_dirs = ["./", "data/"]

    # print to terminal
    for root in root_dirs:
        create_tree(root)
        print()

    # write to markdown file
    f = open("tree.md", "w", encoding="utf-8")
    for i, root in enumerate(root_dirs):
        f.write("```bash\n")
        create_tree_file(root, f)
        f.write("```\n")
        if i != len(root_dirs) - 1:
            f.write("\n")
    f.close()

if __name__ == "__main__":
    main()