from opencc import OpenCC
cc = OpenCC('t2s')
text = "九月的風吹過帶著汗味燕味還有某種更原始的味道"
print(f"Original: {text}")
print(f"Simplified: {cc.convert(text)}")
