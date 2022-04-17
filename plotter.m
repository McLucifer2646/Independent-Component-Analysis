filename_inp_1 = 'input1_new.wav';
filename_inp_2 = 'input2_new.wav';
filename_combine = 'Sample_Mixed_Wave_1.wav';

filename_out_1 = 'out2.wav';
filename_out_2 = 'out1.wav';

[y,Fs] = audioread(filename_combine);
[y1,Fs1] = audioread(filename_inp_1);
[y2,Fs2] = audioread(filename_inp_2);
[y3,Fs3] = audioread(filename_out_1);
[y4,Fs4] = audioread(filename_out_2);

info1 = audioinfo(filename_inp_1);
disp(info1);

SampFreq = info1.SampleRate;
t = 0:seconds(1/Fs):seconds(info1.Duration);
t = t(1:end-1);

subplot(4, 1, 1);
plotter(t, y1, 'Input 1 Signal');

subplot(4, 1, 2);
plotter(t, y2, 'Input 2 Signal');

subplot(4, 1, 3);
plotter(t, y3, 'Output 1 Signal');

subplot(4, 1, 4);
plotter(t, y4, 'Output 2 Signal');

%==============================================================
figure;
subplot(4, 1, 1);
plotter(t, y3, 'Output 1 Signal');

subplot(4, 1, 2);
plotter(t, y4, 'Output 2 Signal');

subplot(4, 1, 3);
my_fft(y3, Fs3, 'Output 1 Signal');

subplot(4, 1, 4);
my_fft(y4, Fs4, 'Output 2 Signal');

%==============================================================
figure;
subplot(2, 1, 1);
plotter(t, y, 'Combined Signal');

subplot(2, 1, 2);
my_fft(y, Fs, 'Combined Signal');

%==============================================================

function my_fft(y_new, Fs, legend_label)
    n = length(y_new)-1;                                            
    f = -Fs/2:Fs/n:Fs/2;
    y_fft = abs(fftshift(fft(y_new)));
    plot(f/1000, y_fft);  
    xlim([-8 8]);
    xlabel ('Frequency(KHz)','fontweight','bold');
    ylabel ('Amplitude','fontweight','bold');
    legend (legend_label)
    title ('Audio Signal plotted in Frequency domain');
    grid on
end

function plotter(t, y_var, legend_label)
    plot(t, y_var);
    xlabel('Time (sec)','fontweight','bold');
    ylabel('Amplitude','fontweight','bold');
    legend(legend_label);
    title('Audio Signal plotted in Time domain');
end