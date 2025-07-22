import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import FullAttention
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.SWTAttention_Family import WaveletEmbedding   
from torch.linalg import svd  # PyTorch 1.8+ 支持

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class SVD_series_decomp(nn.Module): # Actually a kind of PCA?
    """
    基于奇异值分解（SVD）的时间序列分解模块
    支持按时间维度展开或特征维度展开，返回趋势（低秩部分）和季节（残差）成分
    """
    def __init__(self, top_k=5, unfold_method='time'):
        """
        Args:
            top_k: 保留的最大奇异值数量（决定低秩矩阵秩）
            unfold_method: 矩阵展开方式，'time'（按时间展开）或'feature'（按特征展开）
        """
        super(SVD_series_decomp, self).__init__()
        self.top_k = top_k
        self.unfold_method = unfold_method
        assert unfold_method in ['time', 'feature'], "unfold_method must be 'time' or 'feature'"

    def forward(self, x):
        """
        Args:
            x: 输入时间序列，形状为 [B, T, C]（B=批量, T=时间步, C=特征数）
        Returns:
            x_trend: 趋势成分（低秩重构），形状 [B, T, C]
            x_season: 季节成分（残差），形状 [B, T, C]
        """
        B, T, C = x.shape
        if self.unfold_method == 'time':
            # 按时间展开：将每个时间点的特征向量作为矩阵行，形状变为 [B*T, C]
            M = x.reshape(B*T, C)
        else:
            # 按特征展开：将每个特征的时间序列作为矩阵行，形状变为 [C, B*T]
            M = x.permute(2, 0, 1).reshape(C, B*T)
        
        # 执行SVD分解
        U, S, Vt = svd(M, full_matrices=False)  # 快速SVD，返回前min(m,n)个奇异值

        # 保留前top_k个奇异值
        S_k = S[:self.top_k]
        U_k = U[:, :self.top_k]
        Vt_k = Vt[:self.top_k, :]

        # 重构低秩矩阵（趋势成分）
        M_k = U_k @ torch.diag(S_k) @ Vt_k

        # 恢复原始维度
        if self.unfold_method == 'time':
            x_trend = M_k.reshape(B, T, C)
        else:
            x_trend = M_k.reshape(C, B, T).permute(1, 2, 0)  # 转回 [B, T, C]
        
        # 计算季节成分（残差）
        x_season = x - x_trend
        return x_trend, x_season

class SeasonFrequencyProcessor(nn.Module):
    def __init__(self, top_k=5):
        super(SeasonFrequencyProcessor, self).__init__()
        self.top_k = top_k  # 假设configs中有s这个参数

    def forward(self, time_images_season_list):
        new_season_list = []
        for time_images_season in time_images_season_list:
            b ,t ,c ,num_col = time_images_season.size()
            season_matrix = []
            for col in range(num_col):
                x = time_images_season[:, :, :, col]
                xf = torch.fft.rfft(x.permute(0, 2, 1))
                freq = abs(xf)
                freq[0] = 0
                top_k_freq, top_list = torch.topk(freq, self.top_k)
                xf[freq <= top_k_freq.min()] = 0
                x_season = torch.fft.irfft(xf)
                season_matrix.append(x_season.permute(0, 2, 1))
            season_matrix = torch.stack(season_matrix, dim=-1)
            new_season_list.append(season_matrix)
        return new_season_list


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]# b, c, t, k

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

# multi-resolutional decomp

class MultiResolutionalDecomp(nn.Module):
    def __init__(self, configs):
        super(MultiResolutionalDecomp, self).__init__()
        self.avgpool_sizes = configs.avgpool_sizes
        self.mrp_Modules = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=size, stride=1, padding=size // 2)
                for size in self.avgpool_sizes
            ]
        )

    def get_k_for_seq_len(self, seq_len):
        if seq_len == 96: return 6
        elif seq_len == 48: return 4
        elif seq_len == 24: return 2
        elif seq_len == 12: return 2
        else: raise ValueError(f"Unsupported sequence length: {seq_len}")

    def forward(self, x, plot=True):
        seq_len = x.size(1)
        k = self.get_k_for_seq_len(seq_len)
        used_modules = self.mrp_Modules[:k]

        time_image_trend_list = []
        time_image_season_list = []
        fft_results = []

        for i, module in enumerate(used_modules):
            x_rp = module(x.permute(0, 2, 1)).permute(0, 2, 1)
            x_res = x - x_rp
            time_image_trend_list.append(x_rp)
            time_image_season_list.append(x_res)

            # # 对x_rp进行快速傅里叶变换
            # fft_x_rp = torch.fft.fft(x_rp, dim=1)
            # fft_results.append(fft_x_rp)

        time_image_trend_matrix = torch.stack(time_image_trend_list, dim=3)
        time_image_season_matrix = torch.stack(time_image_season_list, dim=3)

        # if plot:
        #     for i, fft_result in enumerate(fft_results):
        #         fft_result = fft_result.detach().cpu().numpy()[0, :, 0]
        #         frequencies = np.fft.fftfreq(len(fft_result))
        #         magnitudes = np.abs(fft_result)

        #         plt.figure()
        #         plt.plot(frequencies, magnitudes)
        #         plt.title(f"FFT Result for Pooling Size {self.avgpool_sizes[i]}")
        #         plt.xlabel('Frequency')
        #         plt.ylabel('Magnitude')
        #         plt.grid(True)
        #         file_name = f"D:/Code/TimeMixer-main-414/models/decomposition_plots/fft_result_pooling_size_{self.avgpool_sizes[i]}.png"                    
        #         plt.savefig(file_name)
        #         plt.show()

        return time_image_trend_matrix, time_image_season_matrix
        
# k-wise Attention
class KDimSelfAttention(nn.Module):
    def __init__(self):
        super(KDimSelfAttention, self).__init__()
        self.linear_layers = nn.ModuleDict()  # 动态存储不同k_dim的线性层
        self.softmax = nn.Softmax(dim=-1)
    
    def get_linear_layer(self, k_dim, device):
        if str(k_dim) not in self.linear_layers:
            self.linear_layers[str(k_dim)] = nn.ModuleDict({
                'query': nn.Linear(k_dim, k_dim).to(device),
                'key': nn.Linear(k_dim, k_dim).to(device),
                'value': nn.Linear(k_dim, k_dim).to(device)
            })
        return self.linear_layers[str(k_dim)]

    def forward(self, x):
        _, _, _, k_dim = x.size()  # 获取实际k维度
        linears = self.get_linear_layer(k_dim, x.device)  # 获取对应的线性层
        
        q = linears['query'](x)  # [B,C,T,k_dim]
        k = linears['key'](x)    # [B,C,T,k_dim]
        v = linears['value'](x)  # [B,C,T,k_dim]

        attn_scores = torch.einsum('bcti,bcsj->bcts', q, k) / (k_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)
        output = torch.einsum('bcts,bcsj->bctj', attn_weights, v)
        # outp = output.sum(dim = -1)
        # return outp
        return output

# #t-wise Attention
# class BlockSelfAttention(nn.Module):
#     def __init__(self, k_dim):
#         super(BlockSelfAttention, self).__init__()
#         self.query = nn.Linear(k_dim, k_dim)
#         self.key = nn.Linear(k_dim, k_dim)
#         self.value = nn.Linear(k_dim, k_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # 生成查询、键和值
#         q = self.query(x) a
#         k = self.key(x)
#         v = self.value(x)

#         # 计算注意力分数
#         _, _, _, k_dim = x.size()
#         attn_scores = torch.einsum('bcti,bcsj->bcts', q, k) / (k_dim ** 0.5)

#         # 应用 Softmax 函数得到注意力权重
#         attn_weights = self.softmax(attn_scores)

#         # 计算加权和得到输出
#         output = torch.einsum('bcts,bcsj->bctj', attn_weights, v)
#         return output

# class TDimSelfAttention(nn.Module):
#     def __init__(self):
#         super(TDimSelfAttention, self).__init__()

#     def forward(self, x):
#         b, c, t, k = x.size()

#         x = x.reshape(b, c, t // 12, k * 12)
#         x = x.permute(0, 1, 3, 2)

#         block_attn = BlockSelfAttention(t//12).to(x.device)
#         output = block_attn(x)

#         # 恢复原始维度
#         output = output.permute(0, 1, 3, 2)
#         output = output.reshape(b, c, t, k)
#         return output


class TDimSelfAttention(nn.Module):
    """
    d_model: target number of MLP output layer's dim, from subpatch_len to d_model 
    k_independent: if True, use K-independent attention; if False, use K-dependent attention
    """
    def __init__(self, configs, subpatch_size, num_heads=4, k_independent=True):
        super().__init__()
        self.d_model = configs.d_model
        self.subpatch_size = subpatch_size
        self.num_heads = num_heads
        self.k_independent = k_independent
        self.proj_in = nn.Linear(subpatch_size, configs.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(configs.d_model, subpatch_size)

    def forward(self, x, patch_size):
        """
        x: [B, C, T, K]
        patch_size: int
        subpatch_size: int
        """
        if self.k_independent:
            return self._forward_k_independent(x, patch_size)
        else:
            return self._forward_k_dependent(x, patch_size)
    
    def _forward_k_independent(self, x, patch_size):
        """K-scale Independent implementation"""
        B, C, T, K = x.shape
        outs = []

        for k in range(K):
            # 按patch切分，每个patch长度为patch_size，最后一段可能不足
            patches = []
            patch_lens = []
            for start in range(0, T, patch_size):
                end = min(start + patch_size, T)
                patches.append(x[:, :, start:end, k])  # [B, C, patch_len]
                patch_lens.append(end - start)
            patch_num = len(patches)
            
            # 对每个patch再切subpatch
            phase_groups = []
            max_group_len = 0
            group_contents = []  # group_contents[i]: 该phase号所有patch片段（list）
            max_patch_len = max(patch_lens)
            for ph in range(0, max_patch_len, self.subpatch_size):
                group = []  # 该phase号下所有patch的对应子片段
                for p in range(patch_num):
                    this_patch = patches[p]  # [B, C, patch_len]
                    if this_patch.shape[2] > ph: # ph means current phase
                        # 本patch有这一相位
                        sub_end = min(ph + self.subpatch_size, this_patch.shape[2])
                        sub = this_patch[:, :, ph:sub_end]  # [B, C, phase_len_real]
                        
                        # mean-value padding
                        if sub.shape[2] < self.subpatch_size:
                            mean_val = sub.mean(dim=2, keepdim=True)  # [B, C, 1]
                            pad_repeat = self.subpatch_size - sub.shape[2]
                            pad = mean_val.repeat(1, 1, pad_repeat)   # [B, C, pad_repeat]
                            sub = torch.cat([sub, pad], dim=2)        # [B, C, subpatch_size]
                        group.append(sub)
                
                if group:
                    group = torch.cat(group, dim=1)  # 按C和patch合并: [B, C*patch_num, subpatch_size]
                    max_group_len = max(max_group_len, group.shape[1])
                    group_contents.append(group)
            
            # 现有 group_contents, 每个元素 [B, ?(<=C*patch_num), subpatch_size]
            output_groups = []
            for group in group_contents:
                cur_len = group.shape[1]
                if cur_len < max_group_len:
                    mean_val = group.mean(dim=1, keepdim=True)  # [B, 1, subpatch_size]
                    group_pad_repeat = max_group_len - cur_len
                    group_pad = mean_val.repeat(1, group_pad_repeat, 1)     # [B, pad_repeat, subpatch_size]
                    group = torch.cat([group, group_pad], dim=1)      # [B, max_group_len, subpatch_size]
                
                group_embed = self.proj_in(group)
                attn_out, attn_weights = self.attn(group_embed, group_embed, group_embed)
                restored = self.proj_out(attn_out)
                output_groups.append(restored)
                
            outs.append(output_groups)
        
        # 重构结果
        result_list = []
        for k, k_out in enumerate(outs):  # k_out: 当前分量的 phase 个 group
            patch_num = k_out[0].shape[1] // C # max_group_len // C = patch_num
            seq_pieces = []  # 这里收集按原始顺序拼好的片段
            for patch_idx in range(patch_num):
                # 逐patch取phase段
                patch_pieces = []
                for phase_idx, group in enumerate(k_out):
                    # group: [B, max_group_len, subpatch_size]
                    group_reshaped = group.view(B, patch_num, C, self.subpatch_size).permute(0, 2, 1, 3)
                    # [B, C, patch_num, subpatch_size] in current phase
                    patch_piece = group_reshaped[:, :, patch_idx, :]   # [B, C, subpatch_size]
                    patch_pieces.append(patch_piece)
                # 按phase拼起来，恢复patch内部顺序
                patch_seq = torch.cat(patch_pieces, dim=2)  # [B, C, patch_total_len]
                seq_pieces.append(patch_seq)
            # 最后把所有patch拼成全局序列
            k_seq = torch.cat(seq_pieces, dim=2)  # [B, C, 总时序长度]
            result_list.append(k_seq)
        result_x = torch.stack(result_list, dim=-1)
        return result_x
    
    def _forward_k_dependent(self, x, patch_size):
        """K-scale dependent implementation"""
        B, C, T, K = x.shape
        
        # 对所有K维尺度同时进行patch切分
        patches = []
        patch_lens = []
        for start in range(0, T, patch_size):
            end = min(start + patch_size, T)
            patches.append(x[:, :, start:end, :])  # [B, C, patch_len, K]
            patch_lens.append(end - start)
        patch_num = len(patches)
        
        # 对每个patch再切subpatch（相位）
        max_group_len = 0
        group_contents = []  # group_contents[i]: 该phase号所有patch片段（list）
        max_patch_len = max(patch_lens)
        
        for ph in range(0, max_patch_len, self.subpatch_size):
            group = []  # 该phase号下所有patch的对应子片段
            for p in range(patch_num):
                this_patch = patches[p]  # [B, C, patch_len, K]
                if this_patch.shape[2] > ph:  # ph means current phase
                    # 本patch有这一相位
                    sub_end = min(ph + self.subpatch_size, this_patch.shape[2])
                    sub = this_patch[:, :, ph:sub_end, :]  # [B, C, phase_len_real, K]
                    
                    # 为保证后续展平方便，pad到subpatch_size
                    if sub.shape[2] < self.subpatch_size:
                        mean_val = sub.mean(dim=2, keepdim=True)  # [B, C, 1, K]
                        pad_repeat = self.subpatch_size - sub.shape[2]
                        pad = mean_val.repeat(1, 1, pad_repeat, 1)   # [B, C, pad_repeat, K]
                        sub = torch.cat([sub, pad], dim=2)        # [B, C, subpatch_size, K]
                    
                    group.append(sub)
            
            # group: list of [B, C, subpatch_size, K]
            if group:
                # 将所有patch堆叠: [B, C, subpatch_size, K, patch_num]
                group_tensor = torch.stack(group, dim=-1)  # [B, C, subpatch_size, K, patch_num]
                # 将C, K, patch_num三个维度完全展平成一维
                group_flattened = group_tensor.permute(0, 1, 3, 4, 2).reshape(B, C*K*patch_num, self.subpatch_size)
                # 现在: [B, C*K*patch_num, subpatch_size]
                
                max_group_len = max(max_group_len, group_flattened.shape[1])
                group_contents.append(group_flattened)
        
        # 现有 group_contents, 每个元素 [B, C*K*patch_num, subpatch_size]
        output_groups = []
        for group in group_contents:
            cur_len = group.shape[1]
            if cur_len < max_group_len:
                # 取已有片段的均值，按 batch 计算
                mean_val = group.mean(dim=1, keepdim=True)  # [B, 1, subpatch_size]
                group_pad_repeat = max_group_len - cur_len
                group_pad = mean_val.repeat(1, group_pad_repeat, 1)     # [B, pad_repeat, subpatch_size]
                group = torch.cat([group, group_pad], dim=1)      # [B, max_group_len, subpatch_size]
            
            # 应用注意力
            group_embed = self.proj_in(group)
            attn_out, attn_weights = self.attn(group_embed, group_embed, group_embed)
            restored = self.proj_out(attn_out)
            # restored: [B, max_group_len, subpatch_size]
            output_groups.append(restored)
        
        # 重新组装回原始形状
        result_patches = []
        for patch_idx in range(patch_num):
            # 逐patch取phase段
            patch_pieces = []
            for phase_idx, group in enumerate(output_groups):
                # group: [B, max_group_len, subpatch_size]
                # 需要重新reshape回 [B, C, K, patch_num, subpatch_size]
                group_reshaped = group.view(B, C, K, patch_num, self.subpatch_size)
                # 当前patch的当前phase
                patch_piece = group_reshaped[:, :, :, patch_idx, :]   # [B, C, K, subpatch_size]
                patch_pieces.append(patch_piece)
            
            # 按phase拼起来，恢复patch内部顺序
            patch_seq = torch.cat(patch_pieces, dim=3)  # [B, C, K, patch_total_len]
            result_patches.append(patch_seq)
        
        # 最后把所有patch拼成全局序列
        result_x = torch.cat(result_patches, dim=3)  # [B, C, K, 总时序长度]
        # 调整维度顺序回到原始格式
        result_x = result_x.permute(0, 1, 3, 2)  # [B, C, T, K]
        
        return result_x


# c-wise attention
class CDimSelfAttention(nn.Module):
    def __init__(self, c_dim):
        super(CDimSelfAttention, self).__init__()
        self.query = nn.Linear(c_dim, c_dim)
        self.key = nn.Linear(c_dim, c_dim)
        self.value = nn.Linear(c_dim, c_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入形状: (batch_size, k, t, c_dim)
        b, t, c_dim = x.size()

        # 生成查询、键和值
        q = self.query(x)  
        k = self.key(x)    
        v = self.value(x)  

        # 计算注意力分数
        attn_scores = torch.einsum('bqd,bkd->bqk', q, k) / (c_dim ** 0.5)  
        attn_weights = self.softmax(attn_scores)  

        # 计算加权和得到输出
        output = torch.einsum('bqk,bkd->bqd', attn_weights, v)  

        return output


class MLP(nn.Module):
    """
    多层感知机MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # 第一层全连接
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)  
              
        # 第二层全连接
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Layer Normalization
        x = self.layer_norm(x)
        return x

class TCN_T_Dim_Mixed(nn.Module):
    def __init__(self, C, kernel_size=3, dilation=1):
        super(TCN_T_Dim_Mixed, self).__init__()
        self.C = C
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = nn.GELU()
        
        # 不立即初始化卷积层
        self.conv = None
    
    def _init_conv(self, device):
        # 动态初始化卷积层
        self.conv = nn.Conv1d(
            in_channels=self.C,
            out_channels=self.C,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=(self.kernel_size - 1) * self.dilation // 2,
            groups=self.C  # 通道完全独立
        ).to(device)
    
    def forward(self, x):
        # x 形状: [B, T, C, K]
        B, T, C = x.shape
        
        # 延迟初始化
        if self.conv is None:
            self._init_conv(x.device)
        elif self.conv.in_channels != C:
            # 如果K值变化，重新初始化
            self._init_conv(x.device)
        
        # 确保输入通道匹配
        if self.conv.in_channels != C:
            raise ValueError(f"Input channels {C} don't match conv channels {self.conv.in_channels}")
        
        x_merged = x.permute(0, 2, 1)
        out = self.conv(x_merged)
        out = self.activation(out)
        out = out.permute(0, 2, 1)
        return out

class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        #Multi-resolutional decomp
        self.multi_resolutional_decomp = MultiResolutionalDecomp(configs)
        self.swt_decompose = WaveletEmbedding(d_channel=configs.d_model, swt=True, requires_grad=False, wv='sym4', m=3, kernel_size=None)
        self.swt_compose = WaveletEmbedding(d_channel=configs.d_model, swt=False, requires_grad=False, wv='sym4', m=3, kernel_size=None)
        #k-wise attention
        self.KDimSelfAttention = KDimSelfAttention()

        #t-wise attention
        self.TDimSelfAttention = TDimSelfAttention(configs, 12)

        # season frequency processing
        self.season_frequency_processing = SeasonFrequencyProcessor()
        """
        The SVD method has been deprecated !!!
        """
        # SVD c-wise
        self.SVD_c = SVD_series_decomp(top_k=configs.top_k, unfold_method='feature')

        # SVD t-wise
        self.SVD_t = SVD_series_decomp(top_k=configs.top_k, unfold_method='time')

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.mlp = nn.ModuleList([
            MLP(input_dim=length, hidden_dim=1024, output_dim=length, dropout=configs.dropout)
            for length in [self.seq_len, self.seq_len // 2, self.seq_len // 4, self.seq_len // 8]
        ])
        self.TCNs = TCN_T_Dim_Mixed(C=configs.d_model, kernel_size=3, dilation=1)


        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list: # x: [B, T, C]
            _, T, _ = x.size()
            length_list.append(T)
        
        device = x_list[0].device

        # multi-resolutional decomp
        time_trend_list = []
        time_season_list = []
        coeffs_list = []
        # Use swt_decompose instead of multi_resolutional_decomp
        for x in x_list:
            coeffs = self.swt_decompose(x)
            coeffs_list.append(coeffs)

        # Assume the last element is the approximation coefficients (trend)
        time_trend_list = [coeffs[:, :, :, -1] for coeffs in coeffs_list] # trend doesn't contain K dim
        # Assume the rest are detail coefficients (season)
        time_season_list = [coeffs[:, :, :, :-1] for coeffs in coeffs_list]
        # for x in x_list:
        #     time_image_trend, time_image_season = self.multi_resolutional_decomp(x)
        #     time_trend_list.append(time_image_trend)
        #     time_season_list.append(time_image_season)# b, t, c, k
        
        # trend part
        # b, t, c, k
        time_images_list = []
        for time_image in time_trend_list:
            time_image_m = self.TCNs(time_image)  # b, t, c  easier to see trends; form a smoother series
            time_images_list.append(time_image_m)
        

        # c-wise attention
        time_image_cwa_list = []
        for i, time_image in enumerate(time_images_list): # enumerate() contains a list for example, show a mapping relation.
            b, t, c_dim = time_image.size()  # 获取 c_dim b, t, c
            cdim_attention = CDimSelfAttention(c_dim).to(device)  
            time_image_cwa = cdim_attention(time_image).permute(0, 2, 1) # b ,c, t
            time_image_cwa_list.append(time_image_cwa)

        # season part
        time_image_twa_list = []
        for time_image in time_season_list: 
        # for time_image in time_image_twa_list:
            time_image = time_image.permute(0, 2, 1, 3) # b, c, t, k
            time_image_twa = self.TDimSelfAttention(time_image, 48) # b, c, t, k
            time_image_twa_list.append(time_image_twa)

        time_image_kwa_list = []
        for time_image in time_season_list:
            time_image = time_image.permute(0, 2, 1, 3) # b, c, t, k
            time_image_kwa = self.KDimSelfAttention(time_image) # b, c, t, k
            time_image_kwa_list.append(time_image_kwa)

        # unify the k dimension
        out_timk_list = [torch.mean(out_timk, dim=-1) for out_timk in time_image_kwa_list]
        out_timt_list = [torch.mean(out_timt, dim=-1) for out_timt in time_image_twa_list]
        # When KDimAttention returns a sum of K-scale series, comment the above 2 lines
        
        # new multi-scale mixing
        out_timk_list = self.mixing_multi_scale_trend(out_timk_list) # b, c, t -> b, t, c
        out_timt_list = self.mixing_multi_scale_trend(out_timt_list)
        # When KDimAttention returns a sum of K-scale series, comment the above 2 lines
        out_timc_list = self.mixing_multi_scale_trend(time_image_cwa_list)

        out_season_list = []
        for out_timk, out_timt, length in zip(out_timk_list, out_timt_list,
                                                      length_list):
            out = out_timk + out_timt
            out_season_list.append(out[:, :length, :])
        # When KDimAttention returns a sum of K-scale series, comment the above 4 lines


        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_timc_list,
                                                      length_list):
        # for ori, out_season, out_trend, length in zip(x_list, time_image_kwa_list, out_timc_list,
        #                                               length_list):
            out = out_trend + out_season
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature # mask

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)


        # new Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
