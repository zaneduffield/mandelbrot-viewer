from bigfloat import BigFloat


class ComplexBf:
    def __init__(self, real: BigFloat, imag: BigFloat):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if isinstance(other, BigFloat):
            real = self.real + other
            imag = self.imag
        else:
            real = self.real + other.real
            imag = self.imag + other.imag
        return ComplexBf(real, imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, BigFloat):
            real = self.real - other
            imag = self.imag
        else:
            real = self.real - other.real
            imag = self.imag - other.imag
        return ComplexBf(real, imag)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexBf(real, imag)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, ComplexBf) and not isinstance(other, complex):
            real = self.real / other
            imag = self.imag / other
            return ComplexBf(real, imag)
        raise NotImplementedError

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def __str__(self):
        return str((str(self.real), str(self.imag)))

    def abs_2(self):
        r = float(self.real)
        i = float(self.imag)
        return r*r + i*i
