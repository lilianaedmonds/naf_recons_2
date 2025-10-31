import sigpy as sp

class L1WaveletRecon(sp.app.App):
    def __init__(self, ksp, mask, mps, lamda, max_iter, device=sp.cpu_device):
        # Move all arrays to the specified device (CPU or GPU)
        ksp = sp.to_device(ksp, device)
        mask = sp.to_device(mask, device)
        mps = sp.to_device(mps, device)
        
        img_shape = mps.shape[1:]
        
        # All operators will automatically work on the device of the input arrays
        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.NUFFT(ksp.shape, axes=(-1, -2))
        P = sp.linop.Multiply(ksp.shape, mask)
        self.W = sp.linop.Wavelet(img_shape)
        A = P * F * S * self.W.H
        
        proxg = sp.prox.L1Reg(A.ishape, lamda)
        
        # Initialize wavelet coefficients on the device
        with device:
            self.wav = sp.zeros(A.ishape, dtype=np.complex64)
        
        alpha = 1
        def gradf(x):
            return A.H * (A * x - ksp)

        alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg, 
                                    max_iter=max_iter)
        super().__init__(alg)
        
    def _output(self):
        return self.W.H(self.wav)