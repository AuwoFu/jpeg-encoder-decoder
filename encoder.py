import argparse
import os
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree


def quantize(block, component,type=0,block_size=8):
    q = load_quantization_table(component,type,block_size=block_size)
    return (block / q).round().astype(np.int32)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def write_to_file(filepath, dc, ac,imageSize, blocks_count, blocks_col, blocks_row, tables):
    try:
        f = open(filepath, 'w')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "No such directory: {}".format(
                os.path.dirname(filepath))) from e

    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits for 'table_size'
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                '''# without runlength
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)'''

                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 8 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    # 32 bits for 'blocks_count'
    row,col=imageSize
    f.write(uint_to_binstr(row, 32))
    f.write(uint_to_binstr(col, 32))

    f.write(uint_to_binstr(blocks_count, 32))
    f.write(uint_to_binstr(blocks_col, 32))
    f.write(uint_to_binstr(blocks_row, 32))

    t1, t2, t3 = ac.shape
    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])

            # DC
            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))

            '''# AC, W/O run-length
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
            for i in range(t2):
                category = bits_required(ac[b, i, c])
                f.write(ac_table[category])
                f.write(int_to_binstr(ac[b, i, c]))
            '''
            symbols, values = run_length_encode(ac[b, :, c])
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()

def predictive_coding(dc):
    l,c=dc.shape
    new_dc=np.copy(dc)
    for i in range(1,l):
        for j in range(c):
            new_dc[i][j] = dc[i][j] - dc[i - 1][j]

    return new_dc

def encode(input_file, output_file,imageSize, block_size=8,quantize_type=0):
    print(f'start encode with block size={block_size}')

    image = Image.open(input_file)
    # change color space from RGB to YCbCr
    ycbcr = image.convert('YCbCr')

    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]

    # block size: 8x8
    if rows % block_size == cols % block_size == 0:
        blocks_count = rows // block_size * cols // block_size
        blocks_col = cols // block_size
        blocks_row = rows // block_size
        print(blocks_col, blocks_row)
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of 8"))

    # dc is the top-left cell of the block, ac are all the other cells
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, block_size ** 2 - 1, 3), dtype=np.int32)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                # split 8x8 block and center the data range on zero
                # [0, 255] --> [-128, 127] 8bit to
                block = npmat[i:i + block_size, j:j + block_size, k] - 128

                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix,
                                        'lum' if k == 0 else 'chrom',type=quantize_type,block_size=block_size)
                zz = block_to_zigzag(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    # DC: predictive coding
    dc=predictive_coding(dc)

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    '''H_AC_Y = HuffmanTree(flatten(ac[i, :, 0]for i in range(blocks_count)))
    H_AC_C = HuffmanTree(flatten(ac[i, :, j] for i in range(blocks_count) for j in [1, 2]))'''
    H_AC_Y = HuffmanTree(
        flatten(run_length_encode(ac[i, :, 0])[0]
                for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
        flatten(run_length_encode(ac[i, :, j])[0]
                for i in range(blocks_count) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    write_to_file(output_file, dc, ac, imageSize,blocks_count, blocks_col, blocks_row, tables)
    print('encode complete')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the input image")
    parser.add_argument("output", help="path to the output image")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    encode(input_file, output_file)
