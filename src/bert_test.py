# -*- coding: utf-8 -*-


def Test(tokenizer, model, id2genre_dictionary):
    n = 0

    for text, genre_id in test_datasets:
      tokenized_text = tokenizer([text], return_tensors="pt")
      output_id = model(**tokenized_text).logits.argmax().item()
      output_genre = id2genre_dictionary[output_id]
      correct_genre = id2genre_dictionary[genre_id]
    
      print('原文 : '+text)
      print('予測ジャンル : '+output_genre)
      print('正解ジャンル : '+correct_genre)
    
      if output_id == genre_id:
        n += 1
    
    
    print('\n'+'正解率 : '+str(round(n/len(test_datasets) * 100, 3))+' %')

