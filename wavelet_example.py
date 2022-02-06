import pywt
import librosa
import soundfile
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener

y, sr = librosa.load('collected signals/rotation.wav', sr=16000)
(cA, cD) = pywt.dwt(y, 'db1')
print('cA:',cA)

print('cD:',cD)
print(y.shape, cA.shape)
plt.subplot(3,3,1)
plt.plot(cA, label='symmetric_cA')
plt.plot(cD, label='symmetric_cD')
plt.legend()
soundfile.write('collected signals/new_rotation.wav', cA, sr)

(cA, cD) = pywt.dwt(y, 'db1', mode = 'reflect')
plt.subplot(3,3,2)
plt.plot(cA, label='reflect')
plt.legend()
soundfile.write('collected signals/new_rotation_reflect.wav', cA, sr)

(cA, cD) = pywt.dwt(y, 'db1', mode = 'smooth')
plt.subplot(3,3,3)
plt.plot(cA, label='smooth')
plt.legend()
soundfile.write('collected signals/new_rotation_smooth.wav', cA, sr)

(cA, cD) = pywt.dwt(y, 'db1', mode = 'periodic')
plt.subplot(3,3,4)
plt.plot(cA, label='periodic')
plt.legend()
soundfile.write('collected signals/new_rotation_periodic.wav', cA, sr)

filtered_y = wiener(y)
plt.subplot(3,3,5)
plt.plot(filtered_y, label='wiener')
plt.legend()
soundfile.write('collected signals/new_rotation_wiener.wav', filtered_y, sr)

filtered_cA = wiener(cA)
plt.subplot(3,3,6)
plt.plot(filtered_cA, label='wiener_cA')
plt.legend()
soundfile.write('collected signals/new_rotation_periodic_wiener.wav', filtered_cA, sr)

plt.subplot(3,3,7)
plt.plot(y, label = 'original')
plt.legend()

plt.show()
