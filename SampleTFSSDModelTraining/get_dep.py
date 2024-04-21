import os
import subprocess
import argparse

def get_dependencies(directory):
    dependencies = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("import ") or line.startswith("from "):
                            package_name = line.split()[1].split('.')[0]
                            dependencies.add(package_name.strip())
    return dependencies

def save_dependencies_to_file(dependencies, file_path):
    with open(file_path, 'w') as f:
        for dependency in dependencies:
            f.write(dependency + '\n')

def main(directory, output):
    output_file = output
    dependencies = get_dependencies(directory)
    save_dependencies_to_file(dependencies, output_file)
    print(f"Dependencies extracted and saved to '{output_file}' in the specified directory.")

if __name__ == "__main__":
    default_dir = os.getcwd()+'/dependencies.txt'
    # print(default_dir)
    parser = argparse.ArgumentParser(description="Extract dependencies from a Python project.")
    parser.add_argument('--directory', type=str, default=os.getcwd(), help="Directory path of the Python project (default: current directory)")
    parser.add_argument('--output', type=str, default=default_dir, help="Out file of the list of dependence")

    args = parser.parse_args()

    main(args.directory, args.output)
