# File paths
source_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/59k_wholeconv_eot.txt"
source_bot_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/59k_wholeconv_bot.txt"
destination_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/data_generation/human_data.txt"
destination_bot_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/data_generation/bot_data.txt"

def filter_human_lines(source_path, destination_path):
    try:
        with open(source_path, "r") as infile, open(destination_path, "w") as outfile:
            for line in infile:
                # Check if the line starts with "<human>"
                if line.strip().startswith("<human>"):
                    outfile.write(line)
        print(f"Filtered data saved to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def filter_bot_lines(source_path, destination_path):
    try:
        with open(source_path, "r") as infile, open(destination_path, "w") as outfile:
            for line in infile:
                # Check if the line starts with "<bot>"
                if line.strip().startswith("<bot>"):
                    outfile.write(line)
        print(f"Filtered data saved to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
#filter_human_lines(source_file, destination_file)

filter_bot_lines(source_file, destination_bot_file)