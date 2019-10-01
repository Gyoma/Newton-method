import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [ 0.0, 1.5, 3.6, 4.3, 5.0 ]
y = [ 1.732, 0.931, 1.732, 1.327, 1.289 ]

pX = 5.251

data = pd.DataFrame(np.array([yy for yy in y]).T, columns = [ 1 ])
errdata = pd.DataFrame(np.array([0.5e-3 for i in range(len(y))]).T, columns = [ 1 ])

origX = np.linspace(min(x), max(x), 1000)
origY = (origX + 3 * np.cos(origX) ** 2) ** 0.5 - 0.2 * origX

def Newton ( crdsX, crdsY, x ):

	res = 0
	for i in range(1, len(crdsX)):
		mult = 1.0
		for j in range(i):
			mult *= (x - crdsX[j])
		res += data.loc[i, i + 1] * mult
	res += crdsY[0]

	return res

def AbsError ( crdsX, x ):
	res = 0
	for i in range(1, len(crdsX)):
		mult = 1.0
		for j in range(i):
			mult *= abs(x - crdsX[j])
		res += errdata.loc[i , i + 1] * mult
	res += 0.5e-3

	return res

def CreateData ( crdsX, indexes ):
		if len(indexes) == 0:
			return 0
		if len(indexes) == 1:
			return data[1].iloc[indexes[0]]
		else:
			startX = crdsX[indexes[0]]
			endX = crdsX[indexes[-1]]
			lngth = len(indexes)
			val1 = (data.loc[indexes[-1] - 1, lngth - 1] if (data.shape[1] >= lngth and data.loc[indexes[-1] - 1][lngth - 1] is not None)
					else CreateData(crdsX, indexes[:-1]))
			val2 = (data.loc[indexes[-1], lngth - 1] if (data.shape[1] >= lngth and data.loc[indexes[-1], lngth - 1] is not None)
					else CreateData(crdsX, indexes[1:]))
			val = (val2 - val1) / (endX - startX)
		
			if lngth not in data.columns:
				data[lngth] = None

			data.loc[indexes[-1], lngth] = val

			return val

def CreateErrData ( crdsX, indexes ):
		if len(indexes) == 0:
			return 0
		if len(indexes) == 1:
			return 0.5e-3
		else:
			startX = crdsX[indexes[0]]
			endX = crdsX[indexes[-1]]
			lngth = len(indexes)
			val1 = (errdata.loc[indexes[-1] - 1, lngth - 1] if (errdata.shape[1] >= lngth and errdata.loc[indexes[-1] - 1, lngth - 1] is not None)
					else CreateErrData(crdsX, indexes[:-1]))
			val2 = (errdata.loc[indexes[-1], lngth - 1] if (errdata.shape[1] >= lngth and errdata.loc[indexes[-1], lngth - 1] is not None)
					else CreateErrData(crdsX, indexes[1:]))
			val = (val2 + val1) / abs(endX - startX)
		
			if lngth not in errdata.columns:
				errdata[lngth] = None

			errdata.loc[indexes[-1], lngth] = val

			return val

	
CreateData(x, [i for i in range(len(x))])
CreateErrData(x, [i for i in range(len(x))])
print('Data : \n', data, '\n')
print('Error data: \n', errdata, '\n')

apprX = np.copy(origX)
apprY = [Newton(x, y, xx) for xx in apprX]

plt.plot(apprX, apprY, color = 'green')
plt.plot(origX, origY, color = 'blue')
plt.plot(x, y, 'o', color = 'red')
pY = Newton(x, y, pX)
plt.scatter(pX, pY, marker = '*')
plt.legend([ 'Newton', 'Original', 'Dots are given', 'Extra point' ])
plt.xlabel('X')
plt.ylabel('Y')
abs_error = AbsError(x, pX)
plt.title('Abs. error = %.5e and Rel. error = %.5e' % ( abs_error, abs(abs_error / pY) ))
plt.suptitle('Newton scheme')
plt.show()