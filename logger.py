def save_logs(message : str, log_file_path : str = "logs.log"):
    with open(log_file_path, "a") as f:
        try:
            f.write(message + "\n")
        except Exception as e:
            print(f"Error while saving logs: {e}")