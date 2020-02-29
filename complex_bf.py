from bigfloat import BigFloat


class ComplexBf:
    def __init__(self, real: BigFloat, imag: BigFloat):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if not isinstance(other, ComplexBf):
            real = self.real + other
            imag = self.imag
        else:
            real = self.real + other.real
            imag = self.imag + other.imag
        return ComplexBf(real, imag)

    def __sub__(self, other):
        if not isinstance(other, ComplexBf):
            real = self.real - other
            imag = self.imag
        else:
            real = self.real - other.real
            imag = self.imag - other.imag
        return ComplexBf(real, imag)

    def __mul__(self, other):
        if not isinstance(other, ComplexBf):
            real = self.real * other
            imag = self.imag * other
        else:
            real = self.real * other.real - self.imag * other.imag
            imag = 2 * self.real * self.imag
        return ComplexBf(real, imag)

    def __truediv__(self, other):
        if not isinstance(other, ComplexBf):
            real = self.real / other
            imag = self.imag / other
            return ComplexBf(real, imag)
        raise NotImplementedError

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def abs_2(self):
        r = float(self.real)
        i = float(self.imag)
        return r*r + i*i


ComplexBf.__rmul__ = ComplexBf.__mul__
ComplexBf.__radd__ = ComplexBf.__add__
ComplexBf.__rsub__ = ComplexBf.__sub__
ComplexBf.__rtruediv__ = ComplexBf.__truediv__
