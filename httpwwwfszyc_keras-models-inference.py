from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import collections

import re

import gc

import unicodedata

import six

import tensorflow as tf

import numpy as np

import pandas as pd

import os

import sys

import random

import keras

import tensorflow as tf

import json

from keras.optimizers import Adam

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda

from keras.models import Model

import keras.backend as K

K.set_epsilon(1e-7)

import re

import codecs

import sys

import string

import codecs

import numpy as np

import re

import pandas as pd

from tqdm import *

sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')


from keras_bert.keras_bert.bert import get_model

import tokenization 

from keras_bert.keras_bert import get_custom_objects

#from keras_bert.keras_bert.optimizers import AdamWarmup





def bert_get_result():

    maxlen = 220

    bsz=512

    print('begin_build')

    def checkpoint_loader(checkpoint_file):

        def _loader(name):

            return tf.train.load_variable(checkpoint_file, name)

        return _loader





    def load_trained_model_from_checkpoint(config_file,

                                      #checkpoint_file,

                                           training=False,

                                           seq_len=None):

        """Load trained official model from checkpoint.

        :param config_file: The path to the JSON configuration file.

        :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.

        :param training: If training, the whole model will be returned.

                         Otherwise, the MLM and NSP parts will be ignored.

        :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in

                        position embeddings will be sliced to fit the new length.

        :return:

        """

        with open(config_file, 'r') as reader:

            config = json.loads(reader.read())

        if seq_len is None:

            seq_len = config['max_position_embeddings']

        else:

            seq_len = min(seq_len, config['max_position_embeddings'])

        #loader = checkpoint_loader(checkpoint_file)

        model = get_model(

            token_num=config['vocab_size'],

            pos_num=seq_len,

            seq_len=seq_len,

            embed_dim=config['hidden_size'],

            transformer_num=config['num_hidden_layers'],

            head_num=config['num_attention_heads'],

            feed_forward_dim=config['intermediate_size'],

            training=training,

        )

        if not training:

            inputs, outputs = model

            model = keras.models.Model(inputs=inputs, outputs=outputs)



        return model



    def convert_lines(example, max_seq_length,tokenizer):

        max_seq_length -=2

        all_tokens = []

        longer = 0

        for i in tqdm(range(len(example))):

          tokens_aa = tokenizer.tokenize(example[i])

          if len(tokens_aa)>max_seq_length:

            tokens_a = tokens_aa[:int(max_seq_length/2)]+tokens_aa[-int(max_seq_length/2):]

            longer += 1

            one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

            all_tokens.append(one_token)

          else:

            one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_aa+["[SEP]"])+[0] * (max_seq_length - len(tokens_aa))

            all_tokens.append(one_token)

        print(longer)

        return np.array(all_tokens)



    test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')#[:1024]#.sample(512*2)

    #test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

    

      

    ###

    ###

    ## new models

    print('new models')

    symbols_to_delete = '→★©®●ː☆¶）иʿ。ﬂﬁ₁♭年▪←ʒ、（月■⇌ɹˤ³の¤‿عدويهصقناخلىبمغرʀɴשלוםביエンᴵאעכח‐ικξتحكسةفزط‑地谷улкноה歌мυтэпрдˢᵒʳʸᴺʷᵗʰᵉᵘοςתמדףנרךצט成都ех小土》करमा英文レクサス外国人бьыгя不つзц会下有的加大子ツشءʲшчюж戦щ明קљћ我出生天一家新ʁսհןجі‒公美阿ספ白マルハニチロ社ζ和中法本士相信政治堂版っфچیリ事「」シχψմեայինրւդک《ლさようならعدويهصقناخلىبمغرʀɴשלוםביエンᴵאעכח‐ικξتحكسةفزط‑地谷улкноה歌мυтэпрдˢᵒʳʸᴺʷᵗʰᵉᵘοςתמדףנרךצט成都ех小土》करमा英文レクサス外国人бьыгя不つзц会下有的加大子ツشءʲшчюж戦щ明קљћ我出生天一家新ʁսհןجі‒公美阿ספ白マルハニチロ社ζ和中法本士相信政治堂版っфچیリ事「」シχψմեայինրւդک《ლさようなら\n＼🍕\r🐵😑\xa0\ue014≠\t\uf818\uf04a\xad😢🐶❤️☺\uf0e0😜😎👊\u200b\u200e😁أ😍💖̶💵❥━┣┫Е┗Ｏ►👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏᴇᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ✔\x96\x92😋👏😱‼\x81ジ故障➤\u2009🚌͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘☕♡◐║▬💩💯⛽🚄🏼ஜ۩۞😖ᴠ🚲✒➥😟😈═ˌ💪🙏🎯◄🌹😇💔😡\x7f👌ἐὶήὲἀίῃἴ🙄✬ＳＵＰＥＲＨＩＴ😠\ufeff☻\u2028😉😤⛺♍🙂\u3000👮💙😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆✓◾🍻🍽🎶🌺🤔😪\x08؟🐰🐇🐱🙆😨⬅🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚獄℅ВПАН🐾🐕❣😆🔗🚽舞伎🙈😴🏿🤗🇺🇸♫ѕＣＭ⤵🏆🎃😩█▓▒░\u200a🌠🐟💫💰💎\x95🖐🙅⛲🍰⭐🤐👆🙌\u2002💛🙁👀🙊🙉\u2004❧▰▔ᴼᴷ◞▀\x13🚬▂▃▄▅▆▇↙🤓\ue602😵άόέὸ̄😒͝☹➡🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7✋\uf04c\x9f\x10😣⏺̲̅😌🤑́🌏😯😲∙‛Ἰᾶὁ💞🚓◇🔔📚✏🏀👐\u202d💤🍇\ue613豆🏡▷❔❓⁉❗\u202f👠्🇹🇼🌸蔡🌞˚🎲😛˙关系С💋💀🎄💜🤢َِ✨是\x80\x9c\x9d🗑\u2005💃📣👿༼◕༽😰ḷЗ▱￼🤣卖温哥华议降％你失去所钱拿坏税骗🐝¯🎅\x85🍺آإ🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴⌠ИОРФДЯМ✘😝🖑ὐύύ特殊作群╪💨圆园▶ℐ☭✭🏈😺♪🌍⏏ệ🍔🐮🍁☔🍆🍑🌮🌯☠🤦\u200d♂𝓒𝓲𝓿𝓵안영하세요ЖК🍀😫🤤ῦ在了可以说普通话汉语好极🎼🕺☃🍸🥂🗽🎇🎊🆘☎🤠👩✈🖒✌✰❆☙🚪⚲\u2006⚭⚆⬭⬯⏖○‣⚓∎ℒ▙☏⅛✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ｃϖ\u2000үａᴦᎥһͺ\u2007ｓǀ\u2001ɩ℮ｙｅ൦ｌƽ¸ｗｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋∼ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋℳ𝐀𝐥𝐪❄🚶𝙢Ἱ🤘ͦ💸☼패티Ｗ⋆𝙇ᵻ👂👃ɜ🎫\uf0a7БУ🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾͡๏̯﴿⚾⚽Φ₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ🎾👹￦⎌🏒⛸寓养宠物吗🏄🐀🚑🤷操𝒑𝒚𝒐𝑴🤙🐒℃欢迎来到拉斯𝙫⏩☮🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ⚠🦄巨收赢得鬼愤怒要买额ẽ🚗✊🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷❌⭕▸𝗢𝟳𝟱𝟬⦁株式⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊☐☑多伦⚡☄ǫ🐽🎻🎹⛓🏹╭╮🍷🦆为友谊祝贺与其想象对如直接问用自己猜传教没积唯认识基督徒曾经让耶稣复活死怪他但当们聊些题时候例战胜因圣把全结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁＞ʕ̣Δ🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용✞🔫👁┈╱╲▏▕┃╰▊▋╯┳┊☒凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿☝💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ✅☛𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨♩🐎🤞☞🐸💟🎰🌝🛳点击查🍭𝑥𝑦𝑧ＡＮＧＪＢ👣\uf020◔◡🏉💭🎥♀Ξ🐴👨🤳⬆🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲̱ℏ𝑮𝗕𝗴\x91🍒⠀ꜥⲣⲏ╚🐑⏰↺⇤∏鉄件✾◦♬ї💊\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製虚偽屁理屈｜Г𝑩𝑰𝒀𝑺🌤∵∴𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡῖΛΩ⤏🇳𝒙Ձռձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫☜Βώ💢▲ΜΟΝΑΕ🇱♲𝝈↴↳💒⊘▫Ȼ⬇🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎✧😼🕷ｇｏｖｒｎｍｔｉｄｕ２０８ｆｂ＇ｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦∕🌈🔭🐊🐍\uf10aˆ⚜☁ڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜🔼'

    symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@・ω+=”“[]^–>\\°<~•™ˈʊɒ∞§{}·ταɡ|¢`―ɪ£♥´¹≈÷′ɔ€†μ½ʻπδηλσερνʃ±µº¾．»ав⋅¿¬β⇒›¡₂₃γ″«φ⅓„：¥сɑ！−²ʌ¼⁴⁄₄‚‖⊂⅔¨×θ？∩，ɐ₀≥↑↓／√－‰≤'



    isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

    remove_dict = {ord(c):f'' for c in symbols_to_delete}

    

    abbr_mapping = {'ᴀ':'a','ʙ':'b','ᴄ':'c','ᴅ':'d','ᴇ':'e','ꜰ':'f','ɢ':'g','ʜ':'h',

                      'ɪ':'i','ᴊ':'j','ᴋ':'k','ʟ':'l','ᴍ':'m','ɴ':'n','ᴏ':'o','ᴘ':'p',

                      'ǫ':'q','ʀ':'r','ꜱ':'s','ᴛ':'t','ᴜ':'u','ᴠ':'v','ᴡ':'w','ʏ':'y','ᴢ':'z', '\n':' ',

                      'u.s.a.': 'usa', 'u.s.a': 'usa', 'u.s.': 'usa',  ' u.s ': ' usa ','u s of a': 'usa',

                      ' u.k. ': 'uk', ' u.k ': ' uk ', ' yr old ': ' years old ',

                      ' yrs old ': ' years old ',' ph.d ': ' phd ','kim jong-un': 'the president of north korea',

                      '#metoo': 'metoo', 'trumpster': 'trump supporter','trumper': 'trump supporter',

                      'trumpian':'trump supporter','trumpism':'trump supporter',"trump's" : 'trump',

                      ' u r ': ' you are ',  'e.g.': 'for example','i.e.': 'in other words',

                      'et.al': 'elsewhere', 'antisemitic': 'anti-semitic','sb91':'senate bill',                                   

                      }





    contraction_mapping = {

        "'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

        'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

        'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',

        "can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have', 'can;t': 'cannot','can;t;ve': 'cannot have',

        'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

        "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',

        "couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

        'couldn;t;ve': 'could not have','couldn´t': 'could not', 'couldn´t´ve': 'could not have','couldn’t': 'could not',

        'couldn’t’ve': 'could not have', 'could´ve': 'could have',

        'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

        'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',

        'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',

        "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

        'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not',

        'hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not',

        'hasn’t': 'has not', "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not',

        'haven’t': 'have not',"he'd": 'he would', "he'd've": 'he would have',"he'll": 'he will','he´ll': 'he will',

        "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',   

        'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have',    

        'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',

        "how'd": 'how did',"how'll": 'how will',"how's": 'how is','how,d': 'how did','how,ll': 'how will',

        'how,s': 'how is','how;d': 'how did','how;ll': 'how will','how;s': 'how is','how´d': 'how did','how´ll': 'how will',

        'how´s': 'how is','how’d': 'how did','how’ll': 'how will','how’s': 'how is',"i'd": 'i would',"i'll": 'i will',

        "i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will','i,m': 'i am','i,ve': 'i have','i;d': 'i would',

        'i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not','isn,t': 'is not','isn;t': 'is not',

        'isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is', "it's": 'it is',

        'it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will', 'it;s': 'it is',

        'it´d': 'it would','it´ll': 'it will','it´s': 'it is','it’d': 'it would','it’ll': 'it will','it’s': 'it is',

        'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',

        'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us', 'let’s': 'let us',

        "ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not', 'mayn;t': 'may not',

        'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have',

        'might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not',

        'mightn;t': 'might not','mightn´t': 'might not', 'mightn’t': 'might not','might´ve': 'might have',

        'might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

        "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not',

        'must´ve': 'must have','must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not',

        'needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',

        'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',

        "shan't": 'shall not', 'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not',

        'sha´n´t': 'shall not','sha’n’t': 'shall not',"she'd": 'she would',"she'll": 'she will',"she's": 'she is',

        'she,d': 'she would','she,ll': 'she will', 'she,s': 'she is','she;d': 'she would','she;ll': 'she will',

        'she;s': 'she is','she´d': 'she would','she´ll': 'she will', 'she´s': 'she is','she’d': 'she would',

        'she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have',

        'should;ve': 'should have', "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not',

        'shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have', 'should’ve': 'should have',

        "that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

        'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',

        "there'd": 'there had', "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had',

        'there;s': 'there is', 'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',

        "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have','they,d': 'they would',

        'they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will',

        'they;re': 'they are', 'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are',

        'they´ve': 'they have','they’d': 'they would','they’ll': 'they will','they’re': 'they are','they’ve': 'they have',

        "wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not','wasn’t': 'was not',

        "we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',

        'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

        "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not',

        'we´d': 'we would','we´ll': 'we will',    'we´re': 'we are','we´ve': 'we have','we’d': 'we would',

        'we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',

        "what's": 'what is',    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is',

        'what,ve': 'what have','what;ll': 'what will','what;re': 'what are','what;s': 'what is','what;ve': 'what have',

        'what´ll': 'what will', 'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will',

        'what’re': 'what are','what’s': 'what is', 'what’ve': 'what have',"where'd": 'where did',"where's": 'where is',

        'where,d': 'where did','where,s': 'where is','where;d': 'where did','where;s': 'where is','where´d': 'where did',

        'where´s': 'where is','where’d': 'where did','where’s': 'where is', "who'll": 'who will',"who's": 'who is',

        'who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is','who´ll': 'who will','who´s': 'who is',

        'who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

        'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not',

        'wouldn´t': 'would not','wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are',

        'you,d': 'you would','you,ll': 'you will', 'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

        'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would',

        'you’ll': 'you will','you’re': 'you are','´cause': 'because','’cause': 'because',"you've": "you have",

        "could'nt": 'could not',"havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will",

        "i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would","who're": "who are","who've": "who have",

        "why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",

        "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",

        'don""t': "do not","dosen't": "does not", "dosn't": "does not","shoudn't": "should not","that'll": "that will",

        "there'll": "there will","there're": "there are", "this'll": "this all"," u're": " you are", "ya'll": "you all",

        "you'r ": "you are ","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not",

        "dont't": "do not","gov't": "government","i'ma": "i am","is'nt": "is not","‘i":'i',  ":)": ' smile ',

        ":-)": ' smile ','…':'...', '😉': ' wink ', '😂': ' joy ', '😀': ' stuck out tongue ',  

         }



    dirty_dict = {      

                        re.compile( '[^a-zA-Z][uU] of [oO][^a-zA-Z]'): ' you of all ',

                        re.compile('[wW][hH][^a-zA-Z ][^a-zA-Z ][eE]'):'whore' ,                  #  wh**e

                        re.compile('[wW][hH][^a-zA-Z ][rR][eE]'):'whore',                         #  wh*re   

                        re.compile('[wW][^a-zA-Z ][oO][rR][eE]'):'whore',                         #  w*ore  

                        '[wW] h o r e':'whore',

                      #  re.compile('[sS][hH][^a-zA-Z ][tT] '):'shit ',                         #   sh*t_

                        re.compile(' [sS][hH][^a-zA-Z ][tT]'):' shit',                         #   _sh*t

                        re.compile('[sS][hH][*_x][tT] '):'shit ',                            #   sh*t

                        re.compile(' [sS][^a-zA-Z ][^a-zA-Z ][tT]'):' shit',                   #   _s**t

                      #  re.compile('[sS][^a-zA-Z ][^a-zA-Z ][tT] '):'shit ',                   #   s**t_

                        re.compile('[sS][-*_x][-*_x][tT] '):'shit ',                      #   s**t

                      #  re.compile('[sS][hH][^a-zA-Z ][^a-zA-Z ] '):'shit ',                    #   sh**_ 

                        re.compile(' [sS][hH][^a-zA-Z ][^a-zA-Z ]'):' shit',                     #   _sh** 

                      #  re.compile('[sS][^a-zA-Z ][iI][tT] '):'shit ',                         #   s*it_   

                        re.compile(' [sS][^a-zA-Z ][iI][tT]'):' shit',                         #   _s*it 

                        re.compile('[sS][-*_x][iI][tT] '):'shit ',                            #   shit

                        '[sS] h i t':'shit','5h1t': 'shit',

                        re.compile(' [fF][^a-zA-Z ][^a-zA-Z ][kK]'):' fuck',                   #   _f**k

                        re.compile('[fF][^a-zA-Z ][^a-zA-Z ][kK] '):'fuck ',                   #   f**k_

                        re.compile('[fF][-*_x][-*_x][kK]'):'fuck',                       #   f**k

                        re.compile(' [fF][^a-zA-Z ][cC][kK]'):' fuck',                         #   _f*ck

                        re.compile('[fF][^a-zA-Z ][cC][kK] '):'fuck ',                         #   f*ck_

                        re.compile('[fF][-*_x][cC][kK]'):'fuck',                            #   f*ck

                        re.compile(' [fF][uU][^a-zA-Z ][kK]'):' fuck',                         #   _fu*k

                        re.compile('[fF][uU][^a-zA-Z ][kK] '):'fuck ',                         #   fu*k_

                        re.compile('[fF][uU][-*_x][kK]'):'fuck',                            #   fu*k

                        '[pP]huk': 'fuck','[pP]huck': 'fuck','[fF]ukk':'fuck','[fF] u c k':'fuck',

                        '[fF]cuk': 'fuck',' [fF]uks': ' fucks',              

                        re.compile(' [dD][^a-zA-Z ][^a-zA-Z ][kK]'):' dick',                   #   _d**k

                        re.compile('[dD][^a-zA-Z ][^a-zA-Z ][kK] '):'dick ',                   #   d**k_

                        re.compile('[dD][-*_x][-*_x][kK]'):'dick',                       #   d**k

                        re.compile(' [dD][^a-zA-Z ][cC][kK]'):' dick',                         #   _d*ck

                        re.compile('[dD][^a-zA-Z ][cC][kK] '):'dick ',                         #   d*ck_

                        re.compile('[dD][-*_x][cC][kK]'):'dick',                            #   d*ck

                        re.compile(' [dD][iI][^a-zA-Z ][kK]'):' dick',                         #   _di*k

                        re.compile('[dD][iI][^a-zA-Z ][kK] '):'dick ',                         #   di*k_

                        re.compile('[dD][iI][-*_x][kK]'):'dick',                            #   di*k



                        re.compile(' [sS][^a-zA-Z ][cC][kK]'):' suck',                         #   _s*ck

                        re.compile('[sS][^a-zA-Z ][cC][kK] '):'suck ',                         #   s*ck_

                        re.compile('[sS][-*_x][cC][kK]'):'suck',                            #   s*ck

                        re.compile(' [sS][uU][^a-zA-Z ][kK]'):' suck',                         #   _su*k

                        re.compile('[sS][uU][^a-zA-Z ][kK] '):'suck ',                         #   su*k_

                        re.compile('[sS][uU][-*_x][kK]'):'suck',                            #   su*k



                        re.compile(' [cC][^a-zA-Z ][nN][tT]'):' cunt',                         #   _c*nt

                        re.compile('[cC][^a-zA-Z ][nN][tT] '):'cunt ',                         #   c*nt_

                        re.compile('[cC][-*_x][nN][tT]'):'cunt',                            #   c*nt

                        re.compile(' [cC][uU][^a-zA-Z ][tT]'):' cunt',                         #   _cu*t

                        re.compile('[cC][uU][^a-zA-Z ][tT] '):'cunt ',                         #   cu*t_

                        re.compile('[cC][uU][-*_x][tT]'):'cunt',                            #   cu*t



                        re.compile(' [bB][^a-zA-Z ][tT][cC][hH]'):' bitch',                       #   _b*tch

                        re.compile('[bB][^a-zA-Z ][tT][cC][hH] '):'bitch ',                       #   b*tch_

                        re.compile('[bB][-*_x][tT][cC][hH]'):'bitch',                          #   b*tch

                        re.compile(' [bB][iI][^a-zA-Z ][cC][hH]'):' bitch',                       #   _bi*ch

                        re.compile('[bB][iI][^a-zA-Z ][cC][hH] '):'bitch ',                       #   bi*ch_

                        re.compile('[bB][iI][-*_x][cC][hH]'):'bitch',                          #   bi*ch

                        re.compile(' [bB][iI][tT][^a-zA-Z ][hH]'):' bitch',                       #   _bit*h

                        re.compile('[bB][iI][tT][^a-zA-Z ][hH]'):'bitch ',                       #   bit*h_

                        re.compile('[bB][iI][tT][-*_x][hH]'):'bitch',                          #   bit*h

                        re.compile('[bB][^a-zA-Z ][tT][^a-zA-Z ][hH]'):'bitch',                   #   b*t*h

                        'b[-*_x][-*_x][-*_x]h':'bitch',                                          #   b***h

                        '[bB] i t c h':'bitch',

                        re.compile('[aA][*_]s'):'ass',                                #   a*s

                        re.compile('[aA][^a-zA-Z ][^a-zA-Z ][hH][oO][lL][eE]'):'asshole',               #   a**hole

                        re.compile(' [aA][^a-zA-Z ][^a-zA-Z ][hH]'):' assh',                   #   a**h

                        re.compile('[aA][^a-zA-Z ][sS][hH][oO][lL][eE]'):'asshole',                     #   a*shole

                        re.compile('[aA][sS][^a-zA-Z ][hH][oO][lL][eE]'):'asshole',                     #   as*hole

                        ' [aA]s[*]':' ass','[aA] s s': 'ass ','[aA]sswhole': 'ass hole',

                        re.compile('[aA]ssh[^a-zA-Z ]le'):'asshole',                     #   assh*le

                        '[hH] o l e':'hole',

                        '[bB][*]ll': 'bull', 

                        re.compile('[pP][^a-zA-Z ][sS][sS][yY]'):' pussy',                         #   p*ssy

                        re.compile('[pP][uU][^a-zA-Z ][sS][yY]'):' pussy',                         #   pu*sy

                        re.compile('[pP][uU][sS][^a-zA-Z ][yY]'):' pussy',                         #   pus*y

                        re.compile('[pP][uU][^a-zA-Z ][^a-zA-Z ][yY]'):' pussy',                   #   pu**y

                        re.compile('[pP][^a-zA-Z ][^a-zA-Z ][sS][yY]'):' pussy',                   #   p**sy

                        re.compile(' [pP][^a-zA-Z ][^a-zA-Z ][^a-zA-Z ][yY]'):' pussy',            #   _pussy

                        '[pP]ussi': 'pussy', '[pP]ussies': 'pussy','[pP]ussys': 'pussy', 

                        '[jJ]ack[-]off': 'jerk off','[mM]asterbat[*]': 'masterbate','[gG]od[-]dam': 'god damm',



              }





    new_final_mapping = { 'jackoff': 'jerk off','jerkoff':'jerk off','bestial': 'beastial',

                         'bestiality': 'beastiality', 'd1ck': 'dick', 'lmfao': 'laughing my fucking ass off',

                          'masturbate': 'masterbate', 'cashap24':'cash app','nurnie':'pussy',

                         'n1gger': 'nigger', 'nigga': 'nigger', 'niggas': 'niggers',

                         'clickbait':'click with bait','yuge':'huge','outsider77':'outsider',

                         'numbnuts': 'noob nuts', 'orgasms': 'orgasm', 'trudope':'the prime minister of canada',

                          'daesh':'isis', "qur'an":'the central religious text of islam','gofundme':'go fund me',

                         'finicum':'an american spokesman','trumpkins':'trump with pumpkin',

                           'trumpcare':'trump health care','obamacare':'obama health care','trumpy':'trump',

                          'trumpster': 'trump supporter','trumper': 'trump supporter','trumpettes':'trump',

                         'realdonaldtrump':'real donald trump','trumpeteer[s]?':'trump supporter',

                          'trumpian':'trump supporter','trumpism':'trump supporter',"trump[']s" : 'trump',

                         'trumplethinskin':'trump','trumpo':'trump','trumpies':'trump',

                          'kim jong([- ]?un)?': 'the president of north korea','cheetolini':'trump',

                          'trumpland':'trump land','trumpty':'trump','trumpist[s]?':'trump supporter',

                          ' brotherin ':' brother ', 'beyak':'canadian politician',

                          'trudeaus':'prime minister of canada ','shibai':'failure',

                          'tridentinus':'tridentinum','zupta[s]?':'the south african president',

                           'putrumpski':'putin and trump supporter','twitler':'twitter user',

                           'antisemitic': 'anti semitic', 'sb91':'senate bill', 

                            'utmterm':' utm term','fakenews':'fake news',  'thedonald':'the donald',               

                            'washingtontimes':'washington times','garycrum':'gary crum',

                            'rangermc':'car','tfws':'tuition fee waiver','sjw?':'social justice warrior',

                            'koncerned':'concerned','vinis':'vinys','Yᴏᴜ':'you', 'auwe': 'oh no',

                            'bigly':'big league','drump[f]?':'trump','brexit':'british exit',

                            'utilitas':'utilities','justiciaries': 'justiciary','doctrne':'doctrine',

                           'deplorables': 'deplorable','conartist' : 'con-artist','pizzagate':'pizza gate',

                           'theglobeandmail': 'the globe and mail', 'howhat': 'how that', ' coz ':' because ',

                           'civilbeat':'civil beat','gubmit':'submit','financialpost':'financial post',               

                           'theguardian': 'the guardian','shopo':'shop','fentayal': 'fentanyl',

                         'designation-': 'designation ','mutilitated' : 'mutilated','dood-': 'dood ',

                         'irakis' : 'iraki', 'supporter[a-z]?':'supporter',' u ':' you ', 

                        }





    def pre_clean_abbr_words(x, dic = abbr_mapping):

        for word in dic.keys():

            if word in x:

                x = x.replace(word, dic[word])

        return x



    def correct_contraction(x, dic = contraction_mapping):

        for word in dic.keys():

            if word in x:

                x = x.replace(word, dic[word])

        return x





    def clean_dirty_dict(x, dic = dirty_dict):

        for word in dic.keys():

            x = re.sub(word, dic[word],x)

        return x  









    def handle_punctuation(x):

        x = x.translate(remove_dict)

        x = x.translate(isolate_dict)

        return x





    def spacing_punctuation(text): ##clear puncts

        for punc in new_puncts:

            if punc in text:

                text = text.replace(punc, ' ')

        return text



    '''  

    def final_contraction(x, dic = final_mapping):

        for word in dic.keys():

            if word in x:

                x = x.replace(word, dic[word])

        return x

    '''



    def new_final_contraction(x, dic = new_final_mapping):

        for word in dic.keys():

            x = re.sub(word, dic[word],x)

        return x  

    def preprocess(df_comment):



        # lower

        # clean misspellings

        df_comment = df_comment.str.lower()

        df_comment = df_comment.str.replace('[\'\"\(\[\:]?https?:?//[!-z]+',' ')

        df_comment = df_comment.str.replace('[\'\"\(\[\:]?www[.][!-z]+',' ')

        df_comment = df_comment.apply(pre_clean_abbr_words)

        df_comment = df_comment.apply(correct_contraction) 

        df_comment = df_comment.apply(clean_dirty_dict)



        # clean the text

    #    df_comment = df_comment.apply(spacing_punctuation)

        df_comment = df_comment.apply(lambda x:handle_punctuation(x))

        df_comment = df_comment.apply(new_final_contraction)



        return df_comment

    

    

    

    ## firstlarge models

    

    print('bert_large_uncased_wwm')

    BERT_PRETRAINED_DIR = '../input/bertprototype/wwm_uncased_l-24_h-1024_a-16/wwm_uncased_L-24_H-1024_A-16' 

    print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

    dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')  

    tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)

    print('build tokenizer uncased done')

    modelb = load_trained_model_from_checkpoint(config_file,training=True,seq_len=maxlen)

    

    sequence_outputb  = modelb.layers[-6].output

    pool_outputb = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_outputb)

    aux_outputb = Dense(6,activation='sigmoid',name = 'aux_output')(sequence_outputb)

    model2  = Model(inputs=modelb.input, outputs=[pool_outputb,aux_outputb])

    #model2.compile(optimizer=adamwarm,loss='mse')

    

    model2.load_weights('../input/jul2995365ep2bertlarge/95365ep2bertlarge.h5')

    print('load ba models new')

    eval_lines = (preprocess(test_df['comment_text'])).values

    token_input2 = convert_lines(eval_lines,maxlen,tokenizer)

    seg_input2 = np.zeros((token_input2.shape[0],maxlen))

    mask_input2 = np.ones((token_input2.shape[0],maxlen))

    hehe_model4 = (model2.predict([token_input2, seg_input2,mask_input2],verbose=1,batch_size=256))[0]#

    print('bertlarge_wwm_uncased',hehe_model4[:5])

    submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': hehe_model4.flatten()

    })

    submission.to_csv('submission_bertlarge.csv', index=False)

    

    

    ##then base uncased models

    

    print('bert_based_uncased')

    BERT_PRETRAINED_DIR = '../input/bertprototype/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/' 

    print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

    #checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

    #dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')  

    #tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)

    print('build tokenizer uncased done')

    modelb = load_trained_model_from_checkpoint(config_file,training=True,seq_len=maxlen)

    

    sequence_outputb  = modelb.layers[-6].output

    pool_outputb = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_outputb)

    aux_outputb = Dense(6,activation='sigmoid',name = 'aux_output')(sequence_outputb)

    model2  = Model(inputs=modelb.input, outputs=[pool_outputb,aux_outputb])

    #model2.compile(optimizer=adamwarm,loss='mse')

    ##low

    model2.load_weights('../input/final-model-group2/bertuncasedbase_pre_220_95175_ep2.h5')

    print('load bert base uncased models low')

    hehe_model4 = (model2.predict([token_input2, seg_input2,mask_input2],verbose=1,batch_size=256))[0]#

    print('bertbase_uncased_low',hehe_model4[:5])

    submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': hehe_model4.flatten()

    })

    submission.to_csv('submission_bertbase_low.csv', index=False)

    

    ##high

    model2.load_weights('../input/95282bertbaseuncased/95282bertbaseuncased.h5')

    print('load bert base uncased models high')

    hehe_model4 = (model2.predict([token_input2, seg_input2,mask_input2],verbose=1,batch_size=256))[0]#

    print('bertbase_uncased_high',hehe_model4[:5])

    submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': hehe_model4.flatten()

    })

    submission.to_csv('submission_bertbase_high.csv', index=False)

    

    

    ##last base cased models

    ##first def preprocessing

    symbols_to_delete = '→★©®●ː☆¶）иʿ。ﬂﬁ₁♭年▪←ʒ、（月■⇌ɹˤ³の¤‿عدويهصقناخلىبمغرʀɴשלוםביエンᴵאעכח‐ικξتحكسةفزط‑地谷улкноה歌мυтэпрдˢᵒʳʸᴺʷᵗʰᵉᵘοςתמדףנרךצט成都ех小土》करमा英文レクサス外国人бьыгя不つзц会下有的加大子ツشءʲшчюж戦щ明קљћ我出生天一家新ʁսհןجі‒公美阿ספ白マルハニチロ社ζ和中法本士相信政治堂版っфچیリ事「」シχψմեայինրւդک《ლさようならعدويهصقناخلىبمغرʀɴשלוםביエンᴵאעכח‐ικξتحكسةفزط‑地谷улкноה歌мυтэпрдˢᵒʳʸᴺʷᵗʰᵉᵘοςתמדףנרךצט成都ех小土》करमा英文レクサス外国人бьыгя不つзц会下有的加大子ツشءʲшчюж戦щ明קљћ我出生天一家新ʁսհןجі‒公美阿ספ白マルハニチロ社ζ和中法本士相信政治堂版っфچیリ事「」シχψմեայինրւդک《ლさようなら\n＼🍕\r🐵😑\xa0\ue014≠\t\uf818\uf04a\xad😢🐶❤️☺\uf0e0😜😎👊\u200b\u200e😁أ😍💖̶💵❥━┣┫Е┗Ｏ►👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏᴇᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ✔\x96\x92😋👏😱‼\x81ジ故障➤\u2009🚌͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘☕♡◐║▬💩💯⛽🚄🏼ஜ۩۞😖ᴠ🚲✒➥😟😈═ˌ💪🙏🎯◄🌹😇💔😡\x7f👌ἐὶήὲἀίῃἴ🙄✬ＳＵＰＥＲＨＩＴ😠\ufeff☻\u2028😉😤⛺♍🙂\u3000👮💙😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆✓◾🍻🍽🎶🌺🤔😪\x08؟🐰🐇🐱🙆😨⬅🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚獄℅ВПАН🐾🐕❣😆🔗🚽舞伎🙈😴🏿🤗🇺🇸♫ѕＣＭ⤵🏆🎃😩█▓▒░\u200a🌠🐟💫💰💎\x95🖐🙅⛲🍰⭐🤐👆🙌\u2002💛🙁👀🙊🙉\u2004❧▰▔ᴼᴷ◞▀\x13🚬▂▃▄▅▆▇↙🤓\ue602😵άόέὸ̄😒͝☹➡🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7✋\uf04c\x9f\x10😣⏺̲̅😌🤑́🌏😯😲∙‛Ἰᾶὁ💞🚓◇🔔📚✏🏀👐\u202d💤🍇\ue613豆🏡▷❔❓⁉❗\u202f👠्🇹🇼🌸蔡🌞˚🎲😛˙关系С💋💀🎄💜🤢َِ✨是\x80\x9c\x9d🗑\u2005💃📣👿༼◕༽😰ḷЗ▱￼🤣卖温哥华议降％你失去所钱拿坏税骗🐝¯🎅\x85🍺آإ🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴⌠ИОРФДЯМ✘😝🖑ὐύύ特殊作群╪💨圆园▶ℐ☭✭🏈😺♪🌍⏏ệ🍔🐮🍁☔🍆🍑🌮🌯☠🤦\u200d♂𝓒𝓲𝓿𝓵안영하세요ЖК🍀😫🤤ῦ在了可以说普通话汉语好极🎼🕺☃🍸🥂🗽🎇🎊🆘☎🤠👩✈🖒✌✰❆☙🚪⚲\u2006⚭⚆⬭⬯⏖○‣⚓∎ℒ▙☏⅛✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ｃϖ\u2000үａᴦᎥһͺ\u2007ｓǀ\u2001ɩ℮ｙｅ൦ｌƽ¸ｗｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋∼ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋℳ𝐀𝐥𝐪❄🚶𝙢Ἱ🤘ͦ💸☼패티Ｗ⋆𝙇ᵻ👂👃ɜ🎫\uf0a7БУ🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾͡๏̯﴿⚾⚽Φ₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ🎾👹￦⎌🏒⛸寓养宠物吗🏄🐀🚑🤷操𝒑𝒚𝒐𝑴🤙🐒℃欢迎来到拉斯𝙫⏩☮🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ⚠🦄巨收赢得鬼愤怒要买额ẽ🚗✊🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷❌⭕▸𝗢𝟳𝟱𝟬⦁株式⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊☐☑多伦⚡☄ǫ🐽🎻🎹⛓🏹╭╮🍷🦆为友谊祝贺与其想象对如直接问用自己猜传教没积唯认识基督徒曾经让耶稣复活死怪他但当们聊些题时候例战胜因圣把全结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁＞ʕ̣Δ🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용✞🔫👁┈╱╲▏▕┃╰▊▋╯┳┊☒凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿☝💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ✅☛𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨♩🐎🤞☞🐸💟🎰🌝🛳点击查🍭𝑥𝑦𝑧ＡＮＧＪＢ👣\uf020◔◡🏉💭🎥♀Ξ🐴👨🤳⬆🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲̱ℏ𝑮𝗕𝗴\x91🍒⠀ꜥⲣⲏ╚🐑⏰↺⇤∏鉄件✾◦♬ї💊\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製虚偽屁理屈｜Г𝑩𝑰𝒀𝑺🌤∵∴𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡῖΛΩ⤏🇳𝒙Ձռձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫☜Βώ💢▲ΜΟΝΑΕ🇱♲𝝈↴↳💒⊘▫Ȼ⬇🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎✧😼🕷ｇｏｖｒｎｍｔｉｄｕ２０８ｆｂ＇ｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦∕🌈🔭🐊🐍\uf10aˆ⚜☁ڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜🔼'

    symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@・ω+=”“[]^–>\\°<~•™ˈʊɒ∞§{}·ταɡ|¢`―ɪ£♥´¹≈÷′ɔ€†μ½ʻπδηλσερνʃ±µº¾．»ав⋅¿¬β⇒›¡₂₃γ″«φ⅓„：¥сɑ！−²ʌ¼⁴⁄₄‚‖⊂⅔¨×θ？∩，ɐ₀≥↑↓／√－‰≤'

    

    abbr_mapping = {      'ᴀ':'a','ʙ':'b','ᴄ':'c','ᴅ':'d','ᴇ':'e','ꜰ':'f','ɢ':'g','ʜ':'h',

                      'ɪ':'i','ᴊ':'j','ᴋ':'k','ʟ':'l','ᴍ':'m','ɴ':'n','ᴏ':'o','ᴘ':'p',

                      'ǫ':'q','ʀ':'r','ꜱ':'s','ᴛ':'t','ᴜ':'u','ᴠ':'v','ᴡ':'w','ʏ':'y','ᴢ':'z', '\n':' ',

                       ' yr old ': ' years old ',' yrs old ': ' years old ','co₂':'carbon dioxide',

               }     



    regex_mapping = {

                         '[Uu][.][Ss][.][Aa][.]': 'USA', '[Uu][.][Ss][.][Aa]': 'USA',

                          '[Uu][.][Ss][.]': 'USA',  ' [Uu][.][Ss] ': ' USA ','[Uu] [Ss] of [Aa]': 'USA',

                          ' [Uu][.][Kk][.]? ': ' UK ',' [Pp][Hh][.][Dd] ': ' phd ',' [Uu] [Rr] ': ' you are ',

                         '[Ee][.][Gg][.]': 'for example','[Ii][.][Ee][.]': 'in other words',

                          '[Ee][Tt][.][Aa][Ll]': 'elsewhere',"[Gg]ov[']t": "government",

                         '[Tt][Rr][Uu][Mm][Pp]':'trump','[Oo][Bb][Aa][Mm][Aa]':'obama',

                    }





    new_final_mapping = {  

                            'jackoff': 'jerk off','jerkoff':'jerk off','bestial': 'beastial',

                         'bestiality': 'beastiality', 'd1ck': 'dick', 'lmfao': 'laughing my fucking ass off',

                          'masturbate': 'masterbate', 'cashap24':'cash app','nurnie':'pussy',

                         'n1gger': 'nigger', 'nigga': 'nigger', 'niggas': 'niggers',

                         'clickbait':'click with bait','YUGE':'huge','Outsider77':'outsider',

                         'numbnuts': 'noob nuts', 'orgasms': 'orgasm', 'Trudope':'The prime minister of Canada',

                          '[Dd]aesh':'ISIS', "Qur'an":'the central religious text of Islam','gofundme':'go fund me',

                         'Finicum':'an American spokesman','trumpkins':'trump with pumpkin','trumpettes':'trump',

                           'trump[Cc]are':'trump health care','obama[Cc]are':'obama health care','trumpies':'trump',

                          'trumpster': 'trump supporter','trumper': 'trump supporter', 'trumpy':'trump',

                          'trumpian':'trump supporter','trumpism':'trump supporter',"trump[']s" : 'trump',

                          '[Kk]im [Jj]ong([- ][Uu]n)?': 'the president of north korea','Cheetolini':'trump',

                          'trumpland':'trump land','trumpty':'trump','trumpist[s]?':'trump supporter',

                          'trumpeteer[s]?':'trump supporter','trumplethinskin':'trump','trumpo':'trump',

                          'realDonaldtrump':'real Donald trump','[Tt]heDonald':'the Donald',

                          ' brother[Ii]n ':' brother ', 'Beyak':'Canadian politician',

                          'Trudeaus':'Prime Minister of Canada ','shibai':'failure',

                          'Tridentinus':'Tridentinum','[Zz]upta[s]?':'the South African President',

                           '[Pp]utrumpski':'Putin and trump supporter','Twitler':'twitter user',

                           'antisemitic': 'anti semitic', '[Ss][Bb]91':'senate bill', 

                            'utmterm':' utm term','[Ff]ake[Nn]ews':'fake news', 'Pizzagate':'Pizza gate',                 

                            '[Ww]ashingtontimes':'washington times','[Gg]arycrum':'gary crum',

                            'RangerMC':'car','[Tt][Ff][Ww]s':'tuition fee waiver','[Ss][Jj][Ww][Ss]?':'social justice warrior',

                            'Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you', '[Aa]uwe': 'oh no',

                            '[Bb]igly':'big league','Drump[f]?':'Trump','[Bb]rexit':'british exit',

                            'utilitas':'utilities','justiciaries': 'justiciary','doctrne':'doctrine',

                           '[Dd]eplorables': 'deplorable','[Cc][Oo][Nn]artist' : 'con-artist',

                           'theglobeandmail': 'the globe and mail', 'howhat': 'how that', ' coz ':' because ',

                           'civilbeat':'civil beat','gubmit':'submit','financialpost':'financial post',               

                           'theguardian': 'the guardian','shopo':'shop','SHOPO':'shop','fentayal': 'fentanyl',

                         'designation-': 'designation ','[Mm]utilitated' : 'Mutilated','dood-': 'dood ',

                         '[Ii]rakis' : 'iraki', 'supporter[a-z]+':'supporter',' u ':' you ', 

                        }





    contraction_mapping = {

        "'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

        'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

        'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',

        "can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have', 'can;t': 'cannot','can;t;ve': 'cannot have',

        'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

        "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',

        "couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

        'couldn;t;ve': 'could not have','couldn´t': 'could not', 'couldn´t´ve': 'could not have','couldn’t': 'could not',

        'couldn’t’ve': 'could not have', 'could´ve': 'could have',

        'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

        'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',

        'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',

        "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

        'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not',

        'hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not',

        'hasn’t': 'has not', "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not',

        'haven’t': 'have not',"he'd": 'he would', "he'd've": 'he would have',"he'll": 'he will','he´ll': 'he will',

        "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',   

        'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have',    

        'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',

        "how'd": 'how did',"how'll": 'how will',"how's": 'how is','how,d': 'how did','how,ll': 'how will',

        'how,s': 'how is','how;d': 'how did','how;ll': 'how will','how;s': 'how is','how´d': 'how did','how´ll': 'how will',

        'how´s': 'how is','how’d': 'how did','how’ll': 'how will','how’s': 'how is',"i'd": 'i would',"i'll": 'i will',

        "i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will','i,m': 'i am','i,ve': 'i have','i;d': 'i would',

        'i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not','isn,t': 'is not','isn;t': 'is not',

        'isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is', "it's": 'it is',

        'it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will', 'it;s': 'it is',

        'it´d': 'it would','it´ll': 'it will','it´s': 'it is','it’d': 'it would','it’ll': 'it will','it’s': 'it is',

        'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',

        'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us', 'let’s': 'let us',

        "ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not', 'mayn;t': 'may not',

        'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have',

        'might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not',

        'mightn;t': 'might not','mightn´t': 'might not', 'mightn’t': 'might not','might´ve': 'might have',

        'might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

        "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not',

        'must´ve': 'must have','must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not',

        'needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',

        'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',

        "shan't": 'shall not', 'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not',

        'sha´n´t': 'shall not','sha’n’t': 'shall not',"she'd": 'she would',"she'll": 'she will',"she's": 'she is',

        'she,d': 'she would','she,ll': 'she will', 'she,s': 'she is','she;d': 'she would','she;ll': 'she will',

        'she;s': 'she is','she´d': 'she would','she´ll': 'she will', 'she´s': 'she is','she’d': 'she would',

        'she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have',

        'should;ve': 'should have', "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not',

        'shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have', 'should’ve': 'should have',

        "that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

        'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',

        "there'd": 'there had', "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had',

        'there;s': 'there is', 'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',

        "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have','they,d': 'they would',

        'they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will',

        'they;re': 'they are', 'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are',

        'they´ve': 'they have','they’d': 'they would','they’ll': 'they will','they’re': 'they are','they’ve': 'they have',

        "wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not','wasn’t': 'was not',

        "we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',

        'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

        "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not',

        'we´d': 'we would','we´ll': 'we will',    'we´re': 'we are','we´ve': 'we have','we’d': 'we would',

        'we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',

        "what's": 'what is',    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is',

        'what,ve': 'what have','what;ll': 'what will','what;re': 'what are','what;s': 'what is','what;ve': 'what have',

        'what´ll': 'what will', 'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will',

        'what’re': 'what are','what’s': 'what is', 'what’ve': 'what have',"where'd": 'where did',"where's": 'where is',

        'where,d': 'where did','where,s': 'where is','where;d': 'where did','where;s': 'where is','where´d': 'where did',

        'where´s': 'where is','where’d': 'where did','where’s': 'where is', "who'll": 'who will',"who's": 'who is',

        'who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is','who´ll': 'who will','who´s': 'who is',

        'who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

        'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not',

        'wouldn´t': 'would not','wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are',

        'you,d': 'you would','you,ll': 'you will', 'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

        'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would',

        'you’ll': 'you will','you’re': 'you are','´cause': 'because','’cause': 'because',"you've": "you have",

        "could'nt": 'could not',"havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will",

        "i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would","who're": "who are","who've": "who have",

        "why'd": "why would","would've": "would have","y'all": "you all",

        "y'know": "you know","you.i": "you i",

        "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",

        'don""t': "do not","dosen't": "does not", "dosn't": "does not","shoudn't": "should not","that'll": "that will",

        "there'll": "there will","there're": "there are", "this'll": "this all", "ya'll": "you all",

        "you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not",

        "dont't": "do not","i'ma": "i am","is'nt": "is not","‘i":'i',  ":)": ' smile ',";)": ' smile ',

        ":-)": ' smile ',":(": ' sad ','…':'...', '😉': ' wink ', '😂': ' joy ', '😀': ' stuck out tongue ',  

         }



    contraction_mapping1 ={

        "Agains't": 'against', "Ain't": 'am not', 'Ain,t': 'am not', 'Ain;t': 'am not', 'Ain´t': 'am not',

     'Ain’t': 'am not', "Aren't": 'are not', 'Aren,t': 'are not', 'Aren;t': 'are not', 'Aren´t': 'are not',

     'Aren’t': 'are not', "Arn't": 'are not', "C'mon": 'common', "Can't": 'cannot', "Can't've": 'cannot have',

     'Can,t': 'cannot', 'Can,t,ve': 'cannot have', 'Can;t': 'cannot', 'Can;t;ve': 'cannot have',

     'Can´t': 'cannot', 'Can´t´ve': 'cannot have', 'Can’t': 'cannot', 'Can’t’ve': 'cannot have',

     "Could'nt": 'could not', "Could've": 'could have', 'Could,ve': 'could have', 'Could;ve': 'could have',

     "Couldn't": 'could not', "Couldn't've": 'could not have', 'Couldn,t': 'could not',

     'Couldn,t,ve': 'could not have', 'Couldn;t': 'could not', 'Couldn;t;ve': 'could not have',

     'Couldn´t': 'could not', 'Couldn´t´ve': 'could not have', 'Couldn’t': 'could not',

     'Couldn’t’ve': 'could not have', 'Could´ve': 'could have', 'Could’ve': 'could have',

     "D'int": 'did not', "Did'nt": 'did not', "Didn't": 'did not', 'Didn,t': 'did not',

     'Didn;t': 'did not', 'Didn´t': 'did not', 'Didn’t': 'did not', "Din't": 'did not',

     "Doens't": 'does not', "Doesn't": 'does not', 'Doesn,t': 'does not', 'Doesn;t': 'does not',

     'Doesn´t': 'does not', 'Doesn’t': 'does not', 'Don""t': 'do not', "Don't": 'do not',

     'Don,t': 'do not', 'Don;t': 'do not', "Dont't": 'do not', 'Don´t': 'do not',

     'Don’t': 'do not', "Dosen't": 'does not', "Dosn't": 'does not', "Hadn't": 'had not',

     "Hadn't've": 'had not have', 'Hadn,t': 'had not', 'Hadn,t,ve': 'had not have', 'Hadn;t': 'had not',

     'Hadn;t;ve': 'had not have', 'Hadn´t': 'had not', 'Hadn´t´ve': 'had not have', 'Hadn’t': 'had not',

     'Hadn’t’ve': 'had not have', "Hasn't": 'has not', 'Hasn,t': 'has not', 'Hasn;t': 'has not',

     'Hasn´t': 'has not', 'Hasn’t': 'has not', "Haven't": 'have not', 'Haven,t': 'have not',

     'Haven;t': 'have not', 'Haven´t': 'have not', 'Haven’t': 'have not', "Havn't": 'have not',

     "He'd": 'he would', "He'd've": 'he would have', "He'll": 'he will', "He's": 'he is',

     'He,d': 'he would', 'He,d,ve': 'he would have', 'He,ll': 'he will', 'He,s': 'he is',

     'He;d': 'he would', 'He;d;ve': 'he would have', 'He;ll': 'he will', 'He;s': 'he is',

     'Here’s': 'here is', 'He´d': 'he would', 'He´d´ve': 'he would have', 'He´ll': 'he will',

     'He´s': 'he is', 'He’d': 'he would', 'He’d’ve': 'he would have', 'He’ll': 'he will',

     'He’s': 'he is', "How'd": 'how did', "How'll": 'how will', "How's": 'how is', 'How,d': 'how did',

     'How,ll': 'how will', 'How,s': 'how is', 'How;d': 'how did', 'How;ll': 'how will', 'How;s': 'how is',

     'How´d': 'how did', 'How´ll': 'how will', 'How´s': 'how is', 'How’d': 'how did', 'How’ll': 'how will',

     'How’s': 'how is', 'I""m': 'i am', "I'am": 'i am', "I'd": 'i would', "I'l": 'i will',

     "I'll": 'i will', "I'm": 'i am', "I'ma": 'i am', "I'v": 'i have', "I've": 'i have',

     'I,d': 'i would', 'I,ll': 'i will', 'I,m': 'i am', 'I,ve': 'i have', 'I;d': 'i would',

     'I;ll': 'i will', 'I;m': 'i am', 'I;ve': 'i have', "Is'nt": 'is not', "Isn't": 'is not',

     'Isn,t': 'is not', 'Isn;t': 'is not', 'Isn´t': 'is not', 'Isn’t': 'is not', "It'd": 'it would',

     "It'll": 'it will', "It's": 'it is', 'It,d': 'it would', 'It,ll': 'it will', 'It,s': 'it is',

     'It;d': 'it would', 'It;ll': 'it will', 'It;s': 'it is', 'It´d': 'it would', 'It´ll': 'it will',

     'It´s': 'it is', 'It’d': 'it would', 'It’ll': 'it will', 'It’s': 'it is', 'I´d': 'i would',

     'I´ll': 'i will', 'I´m': 'i am', 'I´ve': 'i have', 'I’d': 'i would', 'I’ll': 'i will', 'I’m': 'i am',

     'I’ve': 'i have', "Let's": 'let us', 'Let,s': 'let us', 'Let;s': 'let us', 'Let´s': 'let us',

     'Let’s': 'let us', "Ma'am": 'madam', 'Ma,am': 'madam', 'Ma;am': 'madam', "Mayn't": 'may not',

     'Mayn,t': 'may not', 'Mayn;t': 'may not', 'Mayn´t': 'may not', 'Mayn’t': 'may not',

     'Ma´am': 'madam', 'Ma’am': 'madam', "Might've": 'might have', 'Might,ve': 'might have',

     'Might;ve': 'might have', "Mightn't": 'might not', 'Mightn,t': 'might not', 'Mightn;t': 'might not',

     'Mightn´t': 'might not', 'Mightn’t': 'might not', 'Might´ve': 'might have', 'Might’ve': 'might have',

     "Must've": 'must have', 'Must,ve': 'must have', 'Must;ve': 'must have', "Mustn't": 'must not',

     'Mustn,t': 'must not', 'Mustn;t': 'must not', 'Mustn´t': 'must not', 'Mustn’t': 'must not',

     'Must´ve': 'must have', 'Must’ve': 'must have', "Needn't": 'need not', 'Needn,t': 'need not',

     'Needn;t': 'need not', 'Needn´t': 'need not', 'Needn’t': 'need not', "Oughtn't": 'ought not',

     'Oughtn,t': 'ought not', 'Oughtn;t': 'ought not', 'Oughtn´t': 'ought not', 'Oughtn’t': 'ought not',

     "Sha'n't": 'shall not', 'Sha,n,t': 'shall not', 'Sha;n;t': 'shall not', "Shan't": 'shall not',

     'Shan,t': 'shall not', 'Shan;t': 'shall not', 'Shan´t': 'shall not', 'Shan’t': 'shall not',

     'Sha´n´t': 'shall not', 'Sha’n’t': 'shall not', "She'd": 'she would', "She'll": 'she will',

     "She's": 'she is', 'She,d': 'she would', 'She,ll': 'she will', 'She,s': 'she is', 'She;d': 'she would',

     'She;ll': 'she will', 'She;s': 'she is', 'She´d': 'she would', 'She´ll': 'she will', 'She´s': 'she is',

     'She’d': 'she would', 'She’ll': 'she will', 'She’s': 'she is', "Shoudn't": 'should not',

     "Should've": 'should have', 'Should,ve': 'should have', 'Should;ve': 'should have',

     "Shouldn't": 'should not', 'Shouldn,t': 'should not', 'Shouldn;t': 'should not',

     'Shouldn´t': 'should not', 'Shouldn’t': 'should not', 'Should´ve': 'should have',

     'Should’ve': 'should have', "That'd": 'that would', "That'll": 'that will',

     "That's": 'that is', 'That,d': 'that would', 'That,s': 'that is', 'That;d': 'that would',

     'That;s': 'that is', 'That´d': 'that would', 'That´s': 'that is', 'That’d': 'that would',

     'That’s': 'that is', "There'd": 'there had', "There'll": 'there will', "There're": 'there are',

     "There's": 'there is', 'There,d': 'there had', 'There,s': 'there is', 'There;d': 'there had',

     'There;s': 'there is', 'There´d': 'there had', 'There´s': 'there is', 'There’d': 'there had',

     'There’s': 'there is', "They'd": 'they would', "They'll": 'they will', "They're": 'they are',

     "They've": 'they have', 'They,d': 'they would', 'They,ll': 'they will', 'They,re': 'they are',

     'They,ve': 'they have', 'They;d': 'they would', 'They;ll': 'they will', 'They;re': 'they are',

     'They;ve': 'they have', 'They´d': 'they would', 'They´ll': 'they will', 'They´re': 'they are',

     'They´ve': 'they have', 'They’d': 'they would', 'They’ll': 'they will', 'They’re': 'they are',

     'They’ve': 'they have', "This'll": 'this all', "Wan't": 'want', "Was'nt": 'was not', "Wasn't": 'was not',

     'Wasn,t': 'was not', 'Wasn;t': 'was not', 'Wasn´t': 'was not', 'Wasn’t': 'was not', "We'd": 'we would',

     "We'll": 'we will', "We're": 'we are', "We've": 'we have', 'We,d': 'we would', 'We,ll': 'we will',

     'We,re': 'we are', 'We,ve': 'we have', 'We;d': 'we would', 'We;ll': 'we will', 'We;re': 'we are',

     'We;ve': 'we have', "Weren't": 'were not', 'Weren,t': 'were not', 'Weren;t': 'were not',

     'Weren´t': 'were not', 'Weren’t': 'were not', 'We´d': 'we would', 'We´ll': 'we will',

     'We´re': 'we are', 'We´ve': 'we have', 'We’d': 'we would', 'We’ll': 'we will', 'We’re': 'we are',

     'We’ve': 'we have', "What'll": 'what will', "What're": 'what are', "What's": 'what is',

     "What've": 'what have', 'What,ll': 'what will', 'What,re': 'what are', 'What,s': 'what is',

     'What,ve': 'what have', 'What;ll': 'what will', 'What;re': 'what are', 'What;s': 'what is',

     'What;ve': 'what have', 'What´ll': 'what will', 'What´re': 'what are', 'What´s': 'what is',

     'What´ve': 'what have', 'What’ll': 'what will', 'What’re': 'what are', 'What’s': 'what is',

     'What’ve': 'what have', "Where'd": 'where did', "Where's": 'where is', 'Where,d': 'where did',

     'Where,s': 'where is', 'Where;d': 'where did', 'Where;s': 'where is', 'Where´d': 'where did',

     'Where´s': 'where is', 'Where’d': 'where did', 'Where’s': 'where is', "Who'd": 'who would',

     "Who'll": 'who will', "Who're": 'who are', "Who's": 'who is', "Who've": 'who have',

     'Who,ll': 'who will', 'Who,s': 'who is', 'Who;ll': 'who will', 'Who;s': 'who is',

     'Who´ll': 'who will', 'Who´s': 'who is', 'Who’ll': 'who will', 'Who’s': 'who is',

     "Why'd": 'why would', "Won't": 'will not', 'Won,t': 'will not', 'Won;t': 'will not',

     'Won´t': 'will not', 'Won’t': 'will not', "Would've": 'would have', "Wouldn't": 'would not',

     'Wouldn,t': 'would not', 'Wouldn;t': 'would not', 'Wouldn´t': 'would not', 'Wouldn’t': 'would not',

     "Y'all": 'you all', "Y'know": 'you know', "Ya'll": 'you all', "You'd": 'you would', "You'll": 'you will',

     "You're": 'you are', "You've": 'you have', 'You,d': 'you would', 'You,ll': 'you will', 'You,re': 'you are',

     'You.i': 'you i', 'You;d': 'you would', 'You;ll': 'you will', 'You;re': 'you are',

     "Your'e": 'you are', 'You´d': 'you would', 'You´ll': 'you will', 'You´re': 'you are',

     'You’d': 'you would', 'You’ll': 'you will', 'You’re': 'you are', 'You’ve': 'you have'

    }



    dirty_dict = {      re.compile( '[^a-zA-Z][uU] of [oO][^a-zA-Z]'): ' you of all ',

                        re.compile('[wW][hH][^a-zA-Z ][^a-zA-Z ][eE]'):'whore' ,                  #  wh**e

                        re.compile('[wW][hH][^a-zA-Z ][rR][eE]'):'whore',                         #  wh*re   

                        re.compile('[wW][^a-zA-Z ][oO][rR][eE]'):'whore',                         #  w*ore  

                        '[wW] h o r e':'whore',

                      #  re.compile('[sS][hH][^a-zA-Z ][tT] '):'shit ',                         #   sh*t_

                        re.compile(' [sS][hH][^a-zA-Z ][tT]'):' shit',                         #   _sh*t

                        re.compile('[sS][hH][*_x][tT] '):'shit ',                            #   sh*t

                        re.compile(' [sS][^a-zA-Z ][^a-zA-Z ][tT]'):' shit',                   #   _s**t

                      #  re.compile('[sS][^a-zA-Z ][^a-zA-Z ][tT] '):'shit ',                   #   s**t_

                        re.compile('[sS][-*_x][-*_x][tT] '):'shit ',                      #   s**t

                      #  re.compile('[sS][hH][^a-zA-Z ][^a-zA-Z ] '):'shit ',                    #   sh**_ 

                        re.compile(' [sS][hH][^a-zA-Z ][^a-zA-Z ]'):' shit',                     #   _sh** 

                      #  re.compile('[sS][^a-zA-Z ][iI][tT] '):'shit ',                         #   s*it_   

                        re.compile(' [sS][^a-zA-Z ][iI][tT]'):' shit',                         #   _s*it 

                        re.compile('[sS][-*_x][iI][tT] '):'shit ',                            #   shit

                        '[sS] h i t':'shit','5h1t': 'shit',

                        re.compile(' [fF][^a-zA-Z ][^a-zA-Z ][kK]'):' fuck',                   #   _f**k

                        re.compile('[fF][^a-zA-Z ][^a-zA-Z ][kK] '):'fuck ',                   #   f**k_

                        re.compile('[fF][-*_x][-*_x][kK]'):'fuck',                       #   f**k

                        re.compile(' [fF][^a-zA-Z ][cC][kK]'):' fuck',                         #   _f*ck

                        re.compile('[fF][^a-zA-Z ][cC][kK] '):'fuck ',                         #   f*ck_

                        re.compile('[fF][-*_x][cC][kK]'):'fuck',                            #   f*ck

                        re.compile(' [fF][uU][^a-zA-Z ][kK]'):' fuck',                         #   _fu*k

                        re.compile('[fF][uU][^a-zA-Z ][kK] '):'fuck ',                         #   fu*k_

                        re.compile('[fF][uU][-*_x][kK]'):'fuck',                            #   fu*k

                        '[pP]huk': 'fuck','[pP]huck': 'fuck','[fF]ukk':'fuck','[fF] u c k':'fuck',

                        '[fF]cuk': 'fuck',' [fF]uks': ' fucks',              

                        re.compile(' [dD][^a-zA-Z ][^a-zA-Z ][kK]'):' dick',                   #   _d**k

                        re.compile('[dD][^a-zA-Z ][^a-zA-Z ][kK] '):'dick ',                   #   d**k_

                        re.compile('[dD][-*_x][-*_x][kK]'):'dick',                       #   d**k

                        re.compile(' [dD][^a-zA-Z ][cC][kK]'):' dick',                         #   _d*ck

                        re.compile('[dD][^a-zA-Z ][cC][kK] '):'dick ',                         #   d*ck_

                        re.compile('[dD][-*_x][cC][kK]'):'dick',                            #   d*ck

                        re.compile(' [dD][iI][^a-zA-Z ][kK]'):' dick',                         #   _di*k

                        re.compile('[dD][iI][^a-zA-Z ][kK] '):'dick ',                         #   di*k_

                        re.compile('[dD][iI][-*_x][kK]'):'dick',                            #   di*k



                        re.compile(' [sS][^a-zA-Z ][cC][kK]'):' suck',                         #   _s*ck

                        re.compile('[sS][^a-zA-Z ][cC][kK] '):'suck ',                         #   s*ck_

                        re.compile('[sS][-*_x][cC][kK]'):'suck',                            #   s*ck

                        re.compile(' [sS][uU][^a-zA-Z ][kK]'):' suck',                         #   _su*k

                        re.compile('[sS][uU][^a-zA-Z ][kK] '):'suck ',                         #   su*k_

                        re.compile('[sS][uU][-*_x][kK]'):'suck',                            #   su*k



                        re.compile(' [cC][^a-zA-Z ][nN][tT]'):' cunt',                         #   _c*nt

                        re.compile('[cC][^a-zA-Z ][nN][tT] '):'cunt ',                         #   c*nt_

                        re.compile('[cC][-*_x][nN][tT]'):'cunt',                            #   c*nt

                        re.compile(' [cC][uU][^a-zA-Z ][tT]'):' cunt',                         #   _cu*t

                        re.compile('[cC][uU][^a-zA-Z ][tT] '):'cunt ',                         #   cu*t_

                        re.compile('[cC][uU][-*_x][tT]'):'cunt',                            #   cu*t



                        re.compile(' [bB][^a-zA-Z ][tT][cC][hH]'):' bitch',                       #   _b*tch

                        re.compile('[bB][^a-zA-Z ][tT][cC][hH] '):'bitch ',                       #   b*tch_

                        re.compile('[bB][-*_x][tT][cC][hH]'):'bitch',                          #   b*tch

                        re.compile(' [bB][iI][^a-zA-Z ][cC][hH]'):' bitch',                       #   _bi*ch

                        re.compile('[bB][iI][^a-zA-Z ][cC][hH] '):'bitch ',                       #   bi*ch_

                        re.compile('[bB][iI][-*_x][cC][hH]'):'bitch',                          #   bi*ch

                        re.compile(' [bB][iI][tT][^a-zA-Z ][hH]'):' bitch',                       #   _bit*h

                        re.compile('[bB][iI][tT][^a-zA-Z ][hH]'):'bitch ',                       #   bit*h_

                        re.compile('[bB][iI][tT][-*_x][hH]'):'bitch',                          #   bit*h

                        re.compile('[bB][^a-zA-Z ][tT][^a-zA-Z ][hH]'):'bitch',                   #   b*t*h

                        '[bB] i t c h':'bitch',

                        re.compile('[aA][*_]s'):'ass',                                #   a*s

                        re.compile('[aA][^a-zA-Z ][^a-zA-Z ][hH][oO][lL][eE]'):'asshole',               #   a**hole

                        re.compile(' [aA][^a-zA-Z ][^a-zA-Z ][hH]'):' assh',                   #   a**h

                        re.compile('[aA][^a-zA-Z ][sS][hH][oO][lL][eE]'):'asshole',                     #   a*shole

                        re.compile('[aA][sS][^a-zA-Z ][hH][oO][lL][eE]'):'asshole',                     #   as*hole

                        ' [aA]s[*]':' ass','[aA] s s': 'ass ','[aA]sswhole': 'ass hole',

                        re.compile('[aA]ssh[^a-zA-Z ]le'):'asshole',                     #   assh*le

                        '[hH] o l e':'hole',

                        '[bB][*]ll': 'bull', 

                        re.compile('[pP][^a-zA-Z ][sS][sS][yY]'):' pussy',                         #   p*ssy

                        re.compile('[pP][uU][^a-zA-Z ][sS][yY]'):' pussy',                         #   pu*sy

                        re.compile('[pP][uU][sS][^a-zA-Z ][yY]'):' pussy',                         #   pus*y

                        re.compile('[pP][uU][^a-zA-Z ][^a-zA-Z ][yY]'):' pussy',                   #   pu**y

                        re.compile('[pP][^a-zA-Z ][^a-zA-Z ][sS][yY]'):' pussy',                   #   p**sy

                        re.compile(' [pP][^a-zA-Z ][^a-zA-Z ][^a-zA-Z ][yY]'):' pussy',            #   _pussy

                        '[pP]ussi': 'pussy', '[pP]ussies': 'pussy','[pP]ussys': 'pussy', 

                        '[jJ]ack[-]off': 'jerk off','[mM]asterbat[*]': 'masterbate','[gG]od[-]dam': 'god damm',

                }

    from nltk.tokenize.treebank import TreebankWordTokenizer

    tokenizer2 = TreebankWordTokenizer()



    isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

    remove_dict = {ord(c):f'' for c in symbols_to_delete}



    def pre_clean_abbr_words(x):

        dic = abbr_mapping

        for word in dic.keys():

            #if word in x:

            x = x.replace(word, dic[word])

        return x



    def clean_regex_words(x):

        dic = regex_mapping

        for word in dic.keys():

            x = re.sub(word, dic[word],x)

        return x  



    def correct_contraction(x):

        dic = contraction_mapping

        for word in dic.keys():

            if word in x:

                x = x.replace(word, dic[word])

        return x



    def correct_contraction1(x):

        dic = contraction_mapping1

        for word in dic.keys():

            if word in x:

                x = x.replace(word, dic[word])

        return x



    def clean_dirty_dict(x):

        dic = dirty_dict

        for word in dic.keys():

            x = re.sub(word, dic[word],x)

        return x  





    def handle_punctuation(x):

        x = x.translate(remove_dict)

        x = x.translate(isolate_dict)

        return x



    def new_final_contraction(x):

        dic = new_final_mapping

        for word in dic.keys():

            x = re.sub(word, dic[word],x)

        return x 



    def handle_contractions(x):

        x = tokenizer2.tokenize(x)

        return x



    def fix_quote(x):

        x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

        x = ' '.join(x)

        return x



    def preprocess(df_comment):



        # lower

        # clean misspellings

        #df_comment = df_comment.str.lower()

        df_comment = df_comment.apply(pre_clean_abbr_words)

        df_comment = df_comment.apply(clean_regex_words)



        df_comment = df_comment.apply(correct_contraction) 

        df_comment = df_comment.apply(correct_contraction1) 

        df_comment = df_comment.apply(clean_dirty_dict)



        # clean the text

        df_comment = df_comment.apply(lambda x:handle_punctuation(x))

        df_comment = df_comment.apply(new_final_contraction)



        df_comment = df_comment.apply(handle_contractions)

        df_comment = df_comment.apply(fix_quote)



        return df_comment

    

    print('bert_based_cased')

    BERT_PRETRAINED_DIR = '../input/bertprototype/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/' 

    print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

    dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')  

    tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=False)

    print('build tokenizer done')

    modelb = load_trained_model_from_checkpoint(config_file,training=True,seq_len=maxlen)

    

    sequence_outputb  = modelb.layers[-6].output

    pool_outputb = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_outputb)

    aux_outputb = Dense(6,activation='sigmoid',name = 'aux_output')(sequence_outputb)

    model2  = Model(inputs=modelb.input, outputs=[pool_outputb,aux_outputb])

    #model2.compile(optimizer=adamwarm,loss='mse')

    #low

    model2.load_weights('../input/final-models-group1/bertcased_pre_220_95089.h5')

    print('load ba models cased')

    eval_lines = (preprocess(test_df['comment_text'])).values

    token_input2 = convert_lines(eval_lines,maxlen,tokenizer)

    print(token_input2[:3])

    hehe_model4 = (model2.predict([token_input2, seg_input2,mask_input2],verbose=1,batch_size=256))[0]#

    print('bertbase_cased_low',hehe_model4[:5])

    submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': hehe_model4.flatten()

    })

    submission.to_csv('submission_bertbase_cased_low.csv', index=False)

    

    #high

    model2.load_weights('../input/final-model-group2/bertcased_pre_220_95108_ep2.h5')

    print('load ba models cased')

    hehe_model4 = (model2.predict([token_input2, seg_input2,mask_input2],verbose=1,batch_size=256))[0]#

    print('bertbase_cased_low',hehe_model4[:5])

    submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': hehe_model4.flatten()

    })

    submission.to_csv('submission_bertbase_cased_high.csv', index=False)

    

    

    K.clear_session()

bert_get_result()

K.clear_session()

print('gpt2')

from keras_gpt_2_latest.keras_gpt_2.loader import load_trained_model_from_checkpoint
bsz = 128

maxlen=300

#model_folder = '../input/gpt2-models/'

config_path = '../input/gpt2hparamsjson/hparams.json'#os.path.join(model_folder, 'hparams.json')

checkpoint_path = 'anything you like, can be a meme.' #os.path.join(model_folder, 'model.ckpt')#can be anything

model = load_trained_model_from_checkpoint(config_path,

                                           checkpoint_path,

                                           seq_len=maxlen,

                                           fixed_input_shape=True)

sequence_output  = model.get_layer(index=-2).output

maxpool_output = keras.layers.GlobalMaxPooling1D()(sequence_output)

avgpool_output = keras.layers.GlobalAveragePooling1D()(sequence_output)

conc_output = keras.layers.concatenate([maxpool_output,avgpool_output])

dropout_output = keras.layers.Dropout(0.4)(conc_output)

real_output = keras.layers.Dense(1,activation='sigmoid',name='real_output')(dropout_output)

aux_output = keras.layers.Dense(6,activation='sigmoid',name='aux_output')(dropout_output)

model2  = keras.models.Model(inputs=model.input, outputs=[real_output,aux_output])



##tokenizing

from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer

import sys

import regex as re

csv_file = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv' # train or test

df = pd.read_csv(csv_file)#[:1024]#.sample(512*2,random_state=112)

df['comment_text'] = df['comment_text'].astype(str)

df["comment_text"] = df["comment_text"].fillna("DUMMY_VALUE")



def tokenize(self, text):

    """ Tokenize a string. """

    bpe_tokens = []

    for token in re.findall(self.pat, text):

        # token = ''.join(self.byte_encoder[ord(b)] for b in token.encode('utf-8'))

        if sys.version_info[0] == 2:

            token = ''.join(self.byte_encoder[ord(b)] for b in token)

        else:

            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))

        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))

    return bpe_tokens



def convert_lines_gpt2(example, max_seq_length, tokenizer):

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        text = re.sub('[ ]+',' ',text)

        tokens_a = tokenizer.tokenize(tokenizer, text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:int(max_seq_length/2)] + tokens_a[-int(max_seq_length/2):]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(tokens_a) + [0]*(max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)



def extract_data_gpt2(

    model_path,

    csv_file,

    dataset,

    max_sequence_length,

    output_path,

):

    os.makedirs(output_path, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_path, cache_dir=None)

    tokenizer.tokenize = tokenize

    sequences = convert_lines_gpt2(df["comment_text"].values, max_sequence_length, tokenizer)

    return sequences



model_path = '../input/gpt2-models'

token_input2 = extract_data_gpt2(model_path,csv_file,dataset='gpt2',max_sequence_length=maxlen,output_path=' ')
model2.load_weights('../input/gpt2-raw-300-tk0-95217-ep2/gpt2_raw_300_tk0_95217_ep2.h5')

hehe_model4 = (model2.predict(token_input2,verbose=1,batch_size=bsz))[0]

print('gpt2',hehe_model4[:5])

submission = pd.DataFrame.from_dict({

'id': df['id'],

'prediction': hehe_model4.flatten()

})

submission.to_csv('submission_gpt2.csv', index=False)
import gc

K.clear_session()

gc.collect()

import pandas as pd

import numpy as np

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

result_bert_df_9 = pd.read_csv('submission_bertlarge.csv')

result_bert_9 = result_bert_df_9['prediction'].values.flatten()

result_bert_df_6 = pd.read_csv('submission_bertbase_low.csv')

result_bert_6 = result_bert_df_6['prediction'].values.flatten()

result_bert_df_7 = pd.read_csv('submission_bertbase_high.csv')

result_bert_7 = result_bert_df_7['prediction'].values.flatten()

result_bert_df_8 = pd.read_csv('submission_bertbase_cased_low.csv')

result_bert_8 = result_bert_df_8['prediction'].values.flatten()

result_bert_df_5 = pd.read_csv('submission_bertbase_cased_high.csv')

result_bert_5 = result_bert_df_5['prediction'].values.flatten()

result_gpt2_df = pd.read_csv('submission_gpt2.csv')

result_gpt2 = result_gpt2_df['prediction'].values.flatten()



result_ensemble = (result_bert_9+result_bert_6+result_bert_7+result_bert_8+result_bert_5+result_gpt2)/6.



submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': result_ensemble

})



submission.to_csv('submission.csv', index=False)