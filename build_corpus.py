"""Формування corpus.txt з тексту повісті Марка Вовчка «Інститутка» (Вікіджерела, видання 1929 р.).

Запуск: python build_corpus.py
Результат: corpus.txt — по одному реченню в рядку (довжина 20-70 символів).
"""
import html
import json
import re
import urllib.parse
import urllib.request

URL = ("https://uk.wikisource.org/w/api.php?action=parse&page="
       + urllib.parse.quote("Інститутка (1929)") + "&prop=text&format=json&formatversion=2")


def fetch_text():
    req = urllib.request.Request(URL, headers={"User-Agent": "lab3-ocr-student-script/1.0"})
    h = json.loads(urllib.request.urlopen(req, timeout=120).read())["parse"]["text"]
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", h, flags=re.S)
    h = re.sub(r"<br\s*/?>|</p>|</div>", "\n", h)
    txt = html.unescape(re.sub(r"<[^>]+>", "", h))
    return re.sub(r"[ \t\xa0]+", " ", txt)


def main():
    txt = fetch_text()
    start = txt.index("Люди дивуються")
    end = txt.find("НАРОДНЯ БІБЛІОТЕКА", start)  # далі йде видавнича реклама
    body = txt[start:end if end > 0 else None]
    with open("institutka.txt", "w", encoding="utf-8") as f:
        f.write(body)
    body = re.sub(r"\s+", " ", body)
    # розбиття на речення за кінцевими знаками
    sents = re.split(r"(?<=[.!?…])\s+", body)
    out, seen = [], set()
    for s in sents:
        s = s.strip(" —-")
        if not (20 <= len(s) <= 70):
            continue
        if re.search(r"[0-9A-Za-z]", s):  # без цифр і латиниці
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("речень у корпусі:", len(out))


if __name__ == "__main__":
    main()
