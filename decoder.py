import argparse
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image



class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32

    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4

    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0

        # the most significant bit indicates the sign of the number
        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_ac_table_without_runLength(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)


    def read_huffman_code(self, table):
        prefix = ''
        # TODO: break the loop if __read_char is not returing new char
        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("size of unsigned int should be greater than 0")
        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    def __int2(self, bin_num):
        return int(bin_num, 2)


def read_image_file(filepath,block_size=8):
    reader = JPEGFileReader(filepath)

    tables = dict()
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()
            #tables[table_name] = reader.read_ac_table_without_runLength()

    row= reader.read_blocks_count()
    col= reader.read_blocks_count()
    image_size=(row,col)

    blocks_count = reader.read_blocks_count()
    blocks_col = reader.read_blocks_count()
    blocks_row = reader.read_blocks_count()

    dc = np.empty((blocks_count, 3), dtype=np.int32)

    AC_count = block_size * block_size - 1
    ac = np.empty((blocks_count, AC_count, 3), dtype=np.int32)

    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']

            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)

            '''# AC W/O run-length coding
            category = reader.read_huffman_code(ac_table)
            for i in range(AC_count):
                ac[block_index,i, component] = reader.read_int(category)'''



            # AC with run-length coding
            # TODO: try to make reading AC coefficients better
            cells_count = 0
            while cells_count < AC_count:
                run_length, size = reader.read_huffman_code(ac_table)

                if (run_length, size) == (0, 0):
                    while cells_count < AC_count:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                else:
                    for i in range(run_length):
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        value = reader.read_int(size)
                        ac[block_index, cells_count, component] = value
                    cells_count += 1
    dc=dePredictive(dc)
    return dc, ac, tables,image_size, blocks_count,blocks_col,blocks_row


def zigzag_to_block(zigzag):
    # assuming that the width and the height of the block are equal
    rows = cols = int(math.sqrt(len(zigzag)))

    if rows * cols != len(zigzag):
        raise ValueError("length of zigzag should be a perfect square")

    block = np.empty((rows, cols), np.int32)

    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]

    return block


def dequantize(block, component,type=0,block_size=8):
    q = load_quantization_table(component,type,block_size=block_size)
    return block * q

def dePredictive(dc):
    l, c = dc.shape
    full_dc=np.copy(dc)
    for i in range(1,l):
        for j in range(c):
            full_dc[i,j]=full_dc[i-1][j]+dc[i][j]
    print(full_dc)
    return full_dc

def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def decode(input,savePath='./example.jpg',block_size=8,quantize_type=0):
    print(f'start decode with block size={block_size}')
    dc, ac, tables, image_size, blocks_count,blocks_col,blocks_row = read_image_file(input,block_size)


    #blocks_per_line = image_side // block_side
    blocks_per_line=blocks_col

    npmat = np.empty((blocks_row*block_size, blocks_col*block_size, 3), dtype=np.uint8)

    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_size
        j = block_index % blocks_per_line * block_size


        for c in range(3):
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom',type=quantize_type,block_size=block_size)
            block = idct_2d(dct_matrix)

            npmat[i:i+block_size, j:j+block_size, c] = block + 128

    # cut off zero padding
    npmat=npmat[:image_size[0],:image_size[1],:]

    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    image.save(savePath)
    #image.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the input image")
    args = parser.parse_args()
    decode(args.input)


