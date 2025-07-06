import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

# text = """Hey there! I just paid $120.50 for a pair of Nike shoes at www.nike.com 😭. Can you believe it? Anyway, I sent the receipt to john.doe@gmail.com — check it ASAP! He’s like, “Why’d u buy that again?” lol 😂. Honestly, I don’t know... maybe bcoz I had a 20% discount-code: SAVE20.

# BTW, let’s meet at 3:30 p.m. near U.S. Embassy (F-8, Islamabad). Oh! Also bring the USB, I left it @your_place last week. FYI: I saw Dr. Sarah O’Neil on CNN’s Health Talk; she said, “Sleep-deprivation causes memory-loss in adults aged 30-40.” Wild, right?

# Final thing — check this repo: https://github.com/jamshed-dev/nlp-practice. And please don’t forget to ping me at 0321-1234567 or msg me via WhatsApp 📲.

# Alright, ttyl bro! 🚀🔥
# """

# doc = nlp(text)

# for token in doc:
#     print([token.text for token in doc])


import pandas as pd

df_spacy = pd.DataFrame({
    "review": [
        "I paid $50 for this! Totally not worth it 😡. Refund please!",
        "Contact me at ahmed123@gmail.com or visit https://mysite.pk for more info.",
        "Dr. Khan said, “It's probably nothing—but we’ll run some tests.”",
        "He scored 98.5% in the final exam!!! 🧠📚 #SmartKid",
        "U.S. policies are often misunderstood in Pakistan, esp. in 2024.",
        "Why’d she say that? I don’t get it… like fr 😭",
        "Meeting @ Jinnah Ave., 3:30 p.m. Bring ID & USB.",
        "My GitHub is github.com/jamshed-dev 💻 — check my repos!",
        "Price went from Rs.1000 to Rs.2000 — insane!",
        "FYI: You can't access the portal after 11:59 PM ⏳"
    ],
    "sentiment": [
        "negative", "neutral", "neutral", "positive", "neutral",
        "negative", "neutral", "positive", "negative", "neutral"
    ]
})

print(df_spacy.head())

# Tokenization and text processing using spaCy
df_spacy['tokens'] = df_spacy['review'].apply(lambda x: [token.text for token in nlp(x)])

# Display the DataFrame with tokens
print(df_spacy[['review', 'tokens']].head())
