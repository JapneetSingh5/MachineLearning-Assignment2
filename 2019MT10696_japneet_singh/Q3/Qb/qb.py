import sys

def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data

def main():
    path_train_data ,path_test_data= process_command()
    # print(path_train_data, path_test_data)

if __name__ == "__main__":
    main()