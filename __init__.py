

# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='vi', repo_id='hexgrad/Kokoro-82M') # <= make sure lang_code matches voice

# This text is for demonstration purposes only, unseen during training
text = '''
- Trang bìa: Chuyến du hành của đứa trẻ tộc Oni, Yamato, chương 28: "Trả lại thanh kiếm và những cô gái mà các người đã bắt cóc ngay!! Lôi Minh Bát Quái!!." Yamato đánh bại Who's Who chỉ bằng một đòn.

'''
# text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
# text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'
# text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
# text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
# text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
# text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
# text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

# 4️⃣ Generate, display, and save audio files in a loop.


while True:
    user_input = input('>>> ')
    generator = pipeline(
        text, voice=user_input, # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        sf.write(f'{i}.wav', audio, 24000) # save each audio file
        sd.play(audio, samplerate=24000)
        sd.wait()
