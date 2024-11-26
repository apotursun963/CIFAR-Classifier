
import numpy as np


# Bu fonksiyon, ortalaması 0 ve varyansı 1 olan standart normal dağılıma göre rastgele sayılar üretir.
# Çıktıdaki değerler genellikle negatif, pozitif veya sıfır olabilir, 
# çünkü normal dağılımın özelliği gereği, sayıların çoğu 0 civarındadır.

normal_dagilim = np.random.randn(2,3)
print(normal_dagilim)

# Bu fonksiyon, 0 ile 1 arasında rastgele sayılar üretir.
# Üretilen değerler her zaman pozitif ve 0 ile 1 arasında olacaktır.
print(np.random.rand(2,3))


# He Initialization He başlatma, özellikle ReLU ve türevleri 
# gibi doğrusal olmayan aktivasyon fonksiyonları için tasarlanmış bir başlatma yöntemidir.
# He başlatma "normal dağılım" kullanır.
# Ancak standart sapma, giriş katmanındaki nöron sayısına bağlı olarak özel olarak ayarlanır.

# he = np.sqrt(2 / "input_size")


# np.maximum:
# bu fonksiyondur ve iki array (veya bir array ile bir skalar) elemanlarını 
# karşılaştırarak her konumda daha büyük olan değeri döndürür

a = np.array([1,2,3])
b = np.array([0,6,1])
print(np.maximum(a, b))
print(np.maximum(0, a))


# np.where(condition, x, y)
# Bu kullanımda, condition koşulunun doğru olduğu yerlerde x, yanlış olduğu yerlerde ise y değeri seçilir.

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, 'Büyük', 'Küçük')
print(result)
