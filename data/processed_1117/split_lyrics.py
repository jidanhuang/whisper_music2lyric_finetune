with open('train_norm.txt', 'r') as f:     # open the input (source) file
    with open('lyric.txt', 'w') as out_f:   # open the output (destination) file for writing
        for i, line in enumerate(f):
            if i >= 3000:       # exit loop after 3000 lines
                break            
            out_f.write(line.split('\t')[-1].strip() + '\n')
        out_f.close()       # close the output file
    f.close()               # close the input file
