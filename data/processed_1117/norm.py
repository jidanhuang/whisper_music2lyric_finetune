import string

with open('train.txt', 'r') as f_in:
    with open('train_norm.txt', 'w') as f_out:
        for line in f_in:
            # # Remove all special characters from the line
            # line_without_special_chars = "".join([char for char in line if char not in string.punctuation])
            
            # # If the line is not empty, write it to the output file
            # if line_without_special_chars.strip() != '':
            #     f_out.write(line_without_special_chars)
            if line.strip() != '' and line.split('\t')[-1].strip()!='':
                f_out.write(line)
    
    # Close the output file
    f_out.close()

# Close the input file
f_in.close()
