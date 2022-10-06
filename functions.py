import pdb

import torch
from torch.autograd import Function
'''
PyTorch 自定义操作torch.autograd.Function
https://zhuanlan.zhihu.com/p/344802526

'''
class VectorQuantization(Function):
    @staticmethod#
    def forward(ctx, inputs, codebook):# inputs.Size([B, num_patches+1, 768]) codebook.Size([512, 256])
        '''
        根据input判断在当前codebook中距离最近的idx的representaion
        '''
        with torch.no_grad():

            # -----------------------------------------------------------------------------------------------
            # step 1 : 将inputs 展开成2D数据 
            embedding_size = codebook.size(1)  # dim=768
            inputs_size = inputs.size()  # inputs.Size([B,num_patches+1, 768])
            inputs_flatten = inputs.view(-1, embedding_size)  # inputs_flatten.Size([B*num_patches+1, 768])
            # -----------------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------------
            # step 2 : 将codebook \ input 在channel上平方后压缩(sum)  变成dim -> 1个值
            codebook_sqr = torch.sum(codebook ** 2, dim=1)# codebook_sqr.Size([512])
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)# inputs_sqr.shape([B*num_patches+1, 1])
            # -----------------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------------
            # step 3 : Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)# distances.Size([B*num_patches+1, 512])
            '''
            分析:(codebook_sqr + inputs_sqr) - 2*(inputs_flatten \mulma codebook.t())
            shape :  [512]   +   [1024, 1]          [1024, 256] \mulma  [256, 512] = ([1024, 512]

            out = beta * mat + alpha * (mat1 \mulma mat2)
            def addmm(self, beta=1, mat, alpha=1, mat1, mat2, out=None)
            
            '''
            # -----------------------------------------------------------------------------------------------

            # -----------------------------------------------------------------------------------------------
            # step 4 : 计算距离最近的idx的representaion
            # 理解: distances.Size([1024, 512]) 1024个 维度为512组成的codebook 
            # 在1024个中各自找到最代表这一个的表征
            _, indices_flatten = torch.min(distances, dim=1)# indices_flatten.Size([B*num_patches+1])
            indices = indices_flatten.view(*inputs_size[:-1])# indices.Size([B,num_patches+1)
            ctx.mark_non_differentiable(indices)# 不知道是干嘛的
            '''
            ctx.mark_non_differentiable():如果输出不可微分的话，通过此功能被告知
            '''
            return indices# indices.Size([16, 8, 8])

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):# inputs.shape([B, N**2+1, 768]) codebook.Size([512, 768])
        '''
        根据input 返回codebook中的(codes, indices_flatten)
        codes
        idx的表征

        codes.Size([16, 8, 8, 256])
        indices_flatten.shape([1024])
        '''

        # -----------------------------------------------------------------------------------------------
        # step 1 : 根据input判断在当前codebook中距离最近的idx的表征
        indices = vq(inputs, codebook)  # indices.Size([B,num_patches+1])
        indices_flatten = indices.view(-1)  # indices_flatten.shape([B*num_patches+1])
        ctx.save_for_backward(indices_flatten, codebook)  # 保存前向传播的输入和输出，为了后面反向传播使用
        ctx.mark_non_differentiable(indices_flatten)

        # -----------------------------------------------------------------------------------------------
        # step 2 : 根据idx的表征 得到响应的codes
        codes_flatten = torch.index_select(codebook, dim=0,  # codes_flatten.Size([B*num_patches+1, 256])
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)  # codes.Size([B,num_patches+1,768)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
