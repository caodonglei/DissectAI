""" Convolutional Layer

    1  1  1  0  0
    0  1  1  0  0       1  0  1       4  3  3
    0  0  1  1  1  ==>  0  1  0  ==>  2  3  3
    0  0  1  1  0       1  0  1       2  3  4
    0  1  1  0  0

    input image 5*5     filter 3*3    output 3*3
                        step=1, bias=0
"""

from .interface import Layer
from .utils import shape_size, matrix_adds
from .errors import ParameterError

class Conv(Layer):
    def __init__(
        self,
        input_shape,
        filter_shape,
        filter_num=1,
        step=1,
    ):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        if len(input_shape) == 3:
            if len(filter_shape) != 3:
                raise ParameterError("filter shape error.")
            if input_shape[2] != filter_shape[2]:
                # the deep of input and filter must be the same
                raise ParameterError(
                    "input shape and filter shape invalid")
        self.filter_num = filter_num
        self.step = step

        self._filters = [
            [0] * shape_size(filter_shape) for _ in range(filter_num)
        ]
        self._biases = [0] * filter_num

    def init_params(self, **kwargs):
        if 'method' in kwargs:
            kwargs['method'](self._filter)
        elif 'weights' in kwargs:
            weights = kwargs['weights']
            if len(self.filter_shape) == 2:
                rows, columns = self.filter_shape
                for num in range(self.filter_num):
                    for i in range(rows):
                        for j in range(columns):
                            self._filters[num][i*columns + j] = weights[num][i][j]

        if 'bias' in kwargs:
            for num in range(self.filter_num):
                self._biases[num] = kwargs['bias'][num]

    def _calc_conv2d(self, x, x_row, x_col, filter_index):
        rows, columns = self.filter_shape
        _filter = self._filters[filter_index]
        sum_filter = 0
        for r in range(rows):
            for c in range(columns):
                sum_filter += x[x_row+r][x_col+c] \
                    * _filter[r*columns+c]
        return sum_filter

    def forward(self, x):
        """ forward calculation for one sample
        """
        step = self.step
        output = []
        if len(self.input_shape) == 2:
            rows, columns = self.input_shape
            frows, fcolumns = self.filter_shape
            for i in range(0, rows - frows + 1, step):
                out_row = []
                for j in range(0, columns - fcolumns + 1, step):
                    all_fiter_sum = 0
                    for num in range(self.filter_num):
                        all_fiter_sum += self._calc_conv2d(x, i, j, num)
                    out_row.append(all_fiter_sum)
                output.append(out_row)
        elif len(self.input_shape) == 3:
            rows, columns, deep = self.input_shape
            frows, fcolumns, _ = self.filter_shape
            for num in range(self.filter_num):
                # output of each filter
                output = []
                for i in range(0, rows - frows + 1, step):
                    out_row = []
                    for j in range(0, columns - fcolumns + 1, step):
                        out_row.append(self._calc_conv2d(x, i, j))
                    output.append(out_row)
                outputs.append(output)
            if len(outputs) == 1:
                return outputs[0]
            return matrix_adds(outputs)
        return output

    def train(self, X, Y):
        pass

    def predict(self, X):
        return [self.forward(x) for x in X]
