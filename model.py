import mxnet as mx
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import Block, nn


class MemN2N(Block):
    def __init__(self, config, **kwargs):
        super(MemN2N, self).__init__(**kwargs)

        self.nwords = config.nwords
        self.init_std = config.init_std
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim

        with self.name_scope():
            self.A = nn.Embedding(self.nwords, self.edim, weight_initializer=init.Normal(self.init_std))
            self.B = nn.Embedding(self.nwords, self.edim, weight_initializer=init.Normal(self.init_std))
            self.C = nn.Dense(self.edim, in_units=self.edim, weight_initializer=init.Normal(self.init_std))

            # Temporal Encoding
            self.T_A = nn.Embedding(self.mem_size, self.edim, weight_initializer=init.Normal(self.init_std))
            self.T_B = nn.Embedding(self.mem_size, self.edim, weight_initializer=init.Normal(self.init_std))

             # Final Predict
            self.W = nn.Dense(self.nwords, in_units=self.edim,weight_initializer=init.Normal(self.init_std))


    def forward(self, x, time, context):
        hid = []
        hid.append(x)
        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = self.A(context)
        Ain_t = self.T_A(time)
        Ain = Ain_c + Ain_t

        # c_i = sum B_ij * u + T_B_i
        Bin_c = self.B(context)
        Bin_t = self.T_B(time)
        Bin = Bin_c + Bin_t
        
        for h in xrange(self.nhop):
            hid3dim = hid[-1].expand_dims(1)
            Aout = nd.batch_dot(hid3dim, Ain.swapaxes(1,2))
            Aout2dim = Aout.reshape((-1, self.mem_size))
            P = nd.softmax(Aout2dim, axis=1)
            
            Prob3dim = P.expand_dims(1)
            Bout = nd.batch_dot(Prob3dim, Bin)
            Bout2dim = Bout.reshape((-1, self.edim))
            
            Cout = self.C(hid[-1])
            Dout = Bout2dim + Cout
            
            if self.lindim == self.edim:
                hid.append(Dout)
            elif self.lindim == 0:
                hid.append(nd.relu(Dout))
            else:
                F = Dout[:, :self.lindim]
                G = Dout[:, self.lindim:]
                K = nd.relu(G)
                hid.append(nd.concat(F, K, dim=1))
        z = self.W(hid[-1])
        return z
