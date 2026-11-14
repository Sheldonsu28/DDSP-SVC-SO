import os
import time
import numpy as np
import torch
import librosa
from logger.saver import Saver
from logger import utils
from torch.amp import autocast, GradScaler

def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10(mean_square_reference / variance_error)
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * gt_mel
    # 计算误差图像
    error_image = pred_mel - test_image_scaled 
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr


def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr

def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -6
    spec_max = 6
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches 
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # forward
            if dtype == torch.float32:
                ddsp_loss, reflow_loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=torch.device(args.device).type, dtype=dtype):
                    ddsp_loss, reflow_loss=model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            
            # handle nan loss
            if torch.isnan(ddsp_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                del reflow_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * ddsp_loss + reflow_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_ddsp_loss, test_reflow_loss = test(args, model, vocoder, loader_test, saver)
                test_loss = args.train.lambda_ddsp * test_ddsp_loss + test_reflow_loss
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                    'validation/ddsp_loss': test_ddsp_loss,
                    'validation/reflow_loss': test_reflow_loss
                })
                
                model.train()


from torch import nn
import torch.nn.functional as F
from nsf_hifigan.nvSTFT import STFT
from torch.amp import autocast, GradScaler
from stylizer.util import vits_kl_loss
from stylizer.stft_loss import MultiResolutionSTFTLoss


def _stylizer_forward_with_decoder_aux(model, *args, **kwargs):
    """Request Q-Exit training losses while retaining older stylizer support."""
    if hasattr(model, 'decoder_exit'):
        outputs = model(*args, **kwargs, return_decoder_aux=True)
        return (outputs[:-1], outputs[-1])
    return (model(*args, **kwargs), {})

def _decoder_auxiliary_loss(args, aux, reference):
    """Reconstruction + compute cost - entropy for the learned exit policy."""
    if not aux:
        return reference.new_zeros(())
    return args.train.get('lambda_decoder_recon', 1.0) * aux['reconstruction_loss'] - args.train.get('lambda_decoder_entropy', 0.01) * aux['entropy'] + args.train.get('lambda_decoder_ponder', 0.001) * aux['expected_steps'].mean() / aux['loop_budget']

def _decoder_log_values(aux):
    if not aux:
        return {}
    return {'train/decoder_reconstruction_loss': aux['reconstruction_loss'].item(), 'train/decoder_exit_entropy': aux['entropy'].item(), 'train/decoder_expected_steps': aux['expected_steps'].mean().item(), 'train/decoder_exit_steps': aux['exit_steps'].float().mean().item()}

def test_style_vc_gan(args, style_model, model, vocoder, loader_test, saver, dtype, sum_mean, unit_mean, unit_var):
    print(' [*] testing...')
    model.eval()
    style_model.eval()
    mrsl = MultiResolutionSTFTLoss(args.device, [(1024, 120, 600, 0.5, 1), (2048, 240, 1200, 0.5, 1), (4096, 480, 2400, 0.5, 1), (512, 50, 240, 0.5, 1)])
    spkc_criterion = nn.CosineEmbeddingLoss()
    test_ddsp_loss = 0.0
    test_reflow_loss = 0.0
    test_ddsp_band_loss = 0.0
    test_rec_loss = 0.0
    test_ddsp_msrl_loss = 0.0
    test_spk_loss = 0.0
    test_formant = 0.0
    test_reflow_style_loss = 0.0
    test_smooth_loss_loss = 0.0
    mel_val_L1_all = 0
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    with torch.no_grad():
        for (bidx, data) in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])
            st_time = time.time()
            (units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask) = style_model(data['units'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'], data['volume'], 0, infer=True)
            mel = model(units, data['f0'], data['volume'], data['spk_id'], vocoder=vocoder, infer=True, aug_shift=formant, return_wav=False, infer_step=args.infer.infer_step, method=args.infer.method, t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
            (ddsp_loss, reflow_loss, _, ddsp_wav, ddsp_mel) = model(units, data['f0'], data['volume'], data['spk_id'], aug_shift=formant, vocoder=vocoder, gt_spec=data['mel'], infer=False, t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_rec_loss += nn.functional.mse_loss(units, data['units']).item()
            test_ddsp_msrl_loss += mrsl(ddsp_wav, data['gt_audio']).item()
            saver.log_spec(data['name'][0], data['mel'], mel)
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            (audio, sr) = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})
            WAV2MEL = STFT(sr=args.data.sampling_rate, n_mels=128, n_fft=2048, win_size=2048, hop_length=512, fmin=40, fmax=22050, clip_val=1e-05)
            audio = audio.unsqueeze(0)
            pre_mel = WAV2MEL.get_mel(signal[0, ...])
            pre_mel = pre_mel.transpose(-1, -2)
            gt_mel = WAV2MEL.get_mel(audio[0, ...])
            gt_mel = gt_mel.transpose(-1, -2)
            if pre_mel.shape[1] != gt_mel.shape[1]:
                gt_mel = gt_mel[:, :pre_mel.shape[1], :]
            saver.log_spec(data['name'][0], gt_mel, pre_mel)
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_L1_all += torch.nn.functional.l1_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
    test_rec_loss /= num_batches
    test_spk_loss /= num_batches
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches
    test_ddsp_msrl_loss /= num_batches
    test_ddsp_band_loss /= num_batches
    test_formant /= num_batches
    test_reflow_style_loss /= num_batches
    test_smooth_loss_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_L1_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_ddsp_band_loss] test_ddsp_band_loss:', test_ddsp_band_loss)
    print(' [test_rec_loss] test_rec_loss:', test_rec_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    print(' Mel Val L1', mel_val_L1_all)
    saver.log_value({'validation/mel_val_mse': mel_val_mse_all})
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({'validation/mel_val_snr': mel_val_snr_all})
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({'validation/mel_val_psnr': mel_val_psnr_all})
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({'validation/mel_val_sisnr': mel_val_sisnr_all})
    saver.log_value({'validation/mel_val_L1': mel_val_L1_all})
    saver.log_value({'validation/test_reflow_style_loss': test_reflow_style_loss})
    return (test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant)

def train_vc_osgan(args, initial_global_step, style_model, model_g, model_d, optimizer, optimizer_d, scheduler, scheduler_d, vocoder, loader_train, loader_test):
    saver = Saver(args, initial_global_step=initial_global_step, expdir=args.env.style_dir)
    params_count = utils.get_network_paras_amount({'model': model_g})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    style_count = utils.get_network_paras_amount({'style': style_model})
    saver.log_info('--- model size ---')
    saver.log_info(style_count)
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model_g = model_g.to(torch.float32)
    model_g.train()
    style_model.train()
    model_d.train()
    for param in model_g.parameters():
        param.requires_grad = False
    saver.log_info('======= start training =======')
    scaler = GradScaler(enabled=args.train.amp_dtype == 'fp16')
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    spkc_criterion = nn.CosineEmbeddingLoss()
    gradient_accumulation_steps = 1
    accumulation_counter = 0
    for epoch in range(start_epoch, args.train.epochs):
        for (batch_idx, data) in enumerate(loader_train):
            if accumulation_counter == 0:
                saver.global_step_increment()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            if dtype == torch.float32:
                (outputs, decoder_aux) = _stylizer_forward_with_decoder_aux(style_model, data['units_ori'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'], None, saver.global_step)
                (units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask) = outputs
                (ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel) = model_g(units, data['f0'], data['volume'], data['spk_id'], aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            else:
                with autocast(device_type=torch.device(args.device).type, dtype=dtype):
                    (outputs, decoder_aux) = _stylizer_forward_with_decoder_aux(style_model, data['units_ori'], data['units_w'], data['units_h'], data['spk_embd'], data['mel'], data['f0'], None, saver.global_step)
                    (units, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, log_det_fwd, z_ps, mask) = outputs
                    (ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel) = model_g(units, data['f0'], data['volume'], data['spk_id'], aug_shift=data['aug_shift'] + formant, vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start)
            spk_loss = 1 * spkc_criterion(data['spk_embd'], spk_pred, torch.ones(spk_pred.size(0), device=args.device))
            transformed_fake = ddsp_wav
            transformed_real = data['gt_audio']
            if dtype == torch.float32:
                (score_loss, feat_loss, loss_d) = model_d.compute_losses(transformed_fake, transformed_real)
            else:
                with autocast(device_type=torch.device(args.device).type, dtype=dtype):
                    (score_loss, feat_loss, loss_d) = model_d.compute_losses(transformed_fake, transformed_real)
            score_loss = score_loss.float()
            feat_loss = feat_loss.float()
            loss_d = loss_d.float()
            units = units.float()
            reconstruction_loss = F.l1_loss(units, data['units_ori'])
            kl_loss = vits_kl_loss(z_fwd, logvar_ps, mu_pr, logvar_pr, log_det_fwd) + vits_kl_loss(z_bkw, logvar_pr, mu_ps, logvar_ps, log_det_bkw) * 0.5
            decoder_loss = _decoder_auxiliary_loss(args, decoder_aux, reconstruction_loss)
            loss = 3.0 * ddsp_loss + reflow_loss + score_loss + feat_loss + reconstruction_loss + spk_loss * 2.0 + kl_loss + decoder_loss
            if torch.isnan(loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                del ddsp_loss
                del reflow_loss
                del score_loss
                del feat_loss
                del ddsp_band_loss
                del reconstruction_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss *= args.train.lambda_ddsp
                d_params = [p for p in model_d.parameters() if p.requires_grad]
                grads_D = torch.autograd.grad(loss_d / gradient_accumulation_steps, d_params, retain_graph=True, create_graph=False, allow_unused=True)
                for (p, g) in zip(d_params, grads_D):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g
                g_params = [p for p in style_model.parameters() if p.requires_grad]
                scaled_loss = scaler.scale(loss / gradient_accumulation_steps) if dtype != torch.float32 else loss / gradient_accumulation_steps
                grads_G = torch.autograd.grad(scaled_loss, g_params, retain_graph=False, create_graph=False, allow_unused=True)
                for (p, g) in zip(g_params, grads_G):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g
                accumulation_counter += 1
                if accumulation_counter >= gradient_accumulation_steps:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                    scheduler_d.step()
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(style_model.parameters(), max_norm=1.0)
                    if dtype != torch.float32:
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        # AMP may skip an overflowing update while calibrating
                        # its scale. Advance the schedule only after a step.
                        if scaler.get_scale() >= previous_scale:
                            scheduler.step()
                    else:
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_d.zero_grad(set_to_none=True)
                    accumulation_counter = 0
            if saver.global_step % args.train.interval_log == 0 and accumulation_counter == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info('epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(epoch, batch_idx, num_batches, args.env.expdir, args.train.interval_log / saver.get_interval_time(), current_lr, loss.item(), saver.get_total_time(), saver.global_step))
                saver.log_value({'train/loss': loss.item(), 'train/ddsp_loss': ddsp_loss.item(), 'train/ddsp_band_loss': ddsp_band_loss.item(), 'train/unit_rec_loss': reconstruction_loss.item(), 'train/reflow_loss': reflow_loss.item(), 'train/kl_loss': kl_loss.item(), 'train/score_loss': score_loss.item(), 'train/d_loss': loss_d.item(), 'train/decoder_loss': decoder_loss.item(), **_decoder_log_values(decoder_aux), 'train/feat_loss': feat_loss, 'train/lr': current_lr})
            if saver.global_step % args.train.interval_val == 0 and accumulation_counter == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                saver.save_gan_model(style_model, model_d, optimizer_save, optimizer_d, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                (test_ddsp_loss, test_reflow_loss, test_ddsp_band_loss, test_rec_loss, test_ddsp_msrl_loss, test_spk_loss, test_formant) = test_style_vc_gan(args, style_model, model_g, vocoder, loader_test, saver, dtype, None, None, None)
                test_loss = args.train.lambda_ddsp * (test_ddsp_loss + test_ddsp_band_loss + test_ddsp_msrl_loss) + test_spk_loss + test_reflow_loss
                saver.log_info(' --- <validation> --- \nloss: {:.3f}. '.format(test_loss))
                saver.log_value({'validation/loss': test_loss, 'validation/ddsp_loss': test_ddsp_loss, 'validation/reflow_loss': test_reflow_loss, 'validation/test_ddsp_band_loss': test_ddsp_band_loss, 'validation/test_ddsp_msrl_loss': test_ddsp_msrl_loss, 'validation/test_rec_loss': test_rec_loss, 'validation/avg_formant': test_formant, 'validation/test_spk_loss': test_spk_loss})
                model_g.train()
                model_d.train()
                style_model.train()
