import numpy as np
import inspect
from engine.error_handle import InvalidField

class H_field_helper:
    def __init__(self, **kwarg) -> None:
        self.ext_H = kwarg.get("EXT_H", None)
        self.demag_H = kwarg.get("Demag", None)

    def external_field(self, t: float, M: np.ndarray, **kwarg):
        if self.ext_H is None:
            return np.array([0, 0, 0], dtype = np.float64)
        ext_H = np.empty(3, dtype = np.float64) # 外部磁場 (A/m)
        for ii in range(3):
            if isinstance(self.ext_H[ii], (int, float)): # time-independent field
                ext_H[ii] = self.ext_H[ii]
            elif callable(self.ext_H[ii]): # time-dependent field
                ext_H[ii] = self.ext_H[ii](t)
            else:
                raise InvalidField(f"Invalid H field at ext_H[{ii}]!")
        return ext_H 
    
    def demagnetization(self, m: np.ndarray, Ms: float):
        pass


if __name__ == "__main__":
    def my_func(x):
        return np.sin(x) + 100
    HH = H_field_helper(EXT_H = [1, my_func, np.sin])
    print(HH.external_field(10))