# %% [markdown]
# # Лабораторна робота №3. Побудова рішення розпізнавання тексту з застосуванням ШІ
#
# **Варіант 5.**
# * Відкритий набір: Perestoroha OCR — рукописні картки з цитатами (Львів, 1605–1606), `lang-uk/perestoroha-ocr`.
# * Інструменти: **T** — Tesseract 5 (`ukr+eng`), **V** — VLM Qwen2.5-VL 7B локально через Ollama.
# * Чинник деградації: якість JPEG, %: 100, 50, 30, 20, 10, 5.
# * Особливість: чи «осучаснює» VLM правопис — окремий CER для історичних графем.
# * Власні дані: фото рукописного тексту з дореформеними літерами + рукописна сторінка з «Інститутки».
# * Текст для синтетичного набору: Марко Вовчок, «Інститутка» (Вікіджерела, видання 1929 р.).

# %% [markdown]
# ## 1. Імпорт бібліотек і персональні параметри

# %%
import os, re, io, json, time, glob, hashlib, unicodedata, warnings, platform, textwrap
from collections import Counter
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pytesseract
import ollama
from PIL import Image, ImageDraw, ImageFont, ImageOps, ExifTags
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---- персональні параметри ----
STUDENT = "Кормило В. В."
GROUP = "ПЗ-47"
VARIANT = 5
STUDENT_ID = 49                 # останні 3 цифри залікової книжки: 049
SEED = 1000 * VARIANT + STUDENT_ID

# окремі генератори для окремих етапів: зміна одного етапу не зсуває випадкові числа інших
rng_data = np.random.default_rng(SEED)        # вибір речень, розмір шрифту
rng_open = np.random.default_rng(SEED + 1)    # вибір карток відкритого набору
rng_boot = np.random.default_rng(SEED + 2)    # бутстреп
rng_plot = np.random.default_rng(SEED + 3)    # розкид точок на графіках
rng_extra = np.random.default_rng(SEED + 4)   # додаткова вибірка (питання 2)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "ukr+eng"
VLM_MODEL = "qwen2.5vl:7b"
FORCE = False   # True — перерахувати розпізнавання, ігноруючи кеш у results/

FONTS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf",
         "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/georgia.ttf"]

for d in ["data/synth", "data/open", "figures", "results", "my_photos", "tmp"]:
    os.makedirs(d, exist_ok=True)

plt.rcParams["figure.dpi"] = 100


def stamp(ax, title):
    """Заголовок графіка з персональним підписом студента."""
    ax.set_title(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)


def savefig(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


# cv2.imread/imwrite у Windows не відкривають шляхи з кирилицею — читаємо через numpy
def imread_gray(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def imread_color(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, enc = cv2.imencode(os.path.splitext(path)[1], img)
    enc.tofile(path)


def sha256_file(path, n=16):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


TESS_VERSION = str(pytesseract.get_tesseract_version())
VLM_DIGEST = next((m.digest for m in ollama.list().models if m.model == VLM_MODEL), None)
print("SEED =", SEED)
print("Python", platform.python_version(), "| Tesseract", TESS_VERSION,
      "| мови:", pytesseract.get_languages(config=""))
print("VLM:", VLM_MODEL, "digest", (VLM_DIGEST or "?")[:12])

# %% [markdown]
# ## 2. Синтетичний набір з еталонною розміткою
#
# 40 речень з «Інститутки» (довжина 20–70 символів), кожне відтворене одним із чотирьох шрифтів
# Windows з кирилицею розміром 22–35 px і збережене на 6 рівнях якості JPEG. Спотворене
# зображення зберігається у PNG (без втрат), тобто артефакти JPEG фіксуються в пікселях.

# %%
with open("corpus.txt", encoding="utf-8") as f:
    corpus = [s.strip() for s in f if 20 <= len(s.strip()) <= 70]

N_LINES = 40
lines = [str(s) for s in rng_data.choice(corpus, size=N_LINES, replace=False)]


def render_line(text, font_path, size=32, pad=12):
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=0)
    return np.array(img)


def degrade(img, kind, level, rng=None):
    out = img.astype(np.float32)
    if kind == "noise" and level > 0:          # σ гаусового шуму
        out = out + rng.normal(0, level, out.shape)
    elif kind == "blur" and level > 0:         # σ гаусового розмиття
        out = cv2.GaussianBlur(out, (0, 0), level)
    elif kind == "jpeg" and level < 100:       # якість JPEG, %
        ok, enc = cv2.imencode(".jpg", out.astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


FACTOR, LEVELS = "jpeg", [100, 50, 30, 20, 10, 5]


def build_synth(texts, prefix, rng, levels):
    recs = []
    for i, text in enumerate(texts):
        font = FONTS[i % len(FONTS)]
        size = int(rng.integers(22, 36))
        clean = render_line(text, font, size=size)
        for lvl in levels:
            path = f"data/synth/{prefix}{i:03d}_{FACTOR}_{lvl}.png"
            imwrite(path, degrade(clean, FACTOR, lvl))
            recs.append({"id": i, "level": lvl, "path": path, "gt": text,
                         "font": os.path.basename(font), "size": size})
    return pd.DataFrame(recs)


ds = build_synth(lines, "", rng_data, LEVELS)
ds.to_csv("data/synth/ground_truth.csv", index=False, lineterminator="\n")
GT_SHA = sha256_file("data/synth/ground_truth.csv")
print("Записів:", len(ds), "| рядків:", N_LINES, "| рівнів:", len(LEVELS), "| SHA-256 еталону:", GT_SHA)
print(ds.drop_duplicates("id")[["id", "font", "size", "gt"]].head(8).to_string())

# %%
fig, axes = plt.subplots(len(LEVELS), 1, figsize=(8, 0.9 * len(LEVELS)))
for ax, lvl in zip(axes, LEVELS):
    img = imread_gray(f"data/synth/000_{FACTOR}_{lvl}.png")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"якість JPEG = {lvl} %", fontsize=8, loc="left")
    ax.axis("off")
fig.suptitle(f"Рядок 0 на шести рівнях якості JPEG\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
savefig(fig, "fig0_samples")

# %% [markdown]
# ## 3. Відкритий набір Perestoroha OCR
#
# Набір містить 430 карток (train 278, validation 122, test 30). Кожна картка — рукописна
# цитата старо-українського тексту 1605–1606 рр. з підписом джерела. Модель не донавчається,
# тому всі три частини об'єднано і випадково (генератор SEED+1) обрано 100 карток.
# Ідентифікатори збережено в `data/open/perestoroha_ids.csv`.

# %%
from datasets import load_dataset, Image as HFImage

hf = load_dataset("lang-uk/perestoroha-ocr").cast_column("image", HFImage(decode=False))
pool = [(split, i) for split in ["train", "validation", "test"] for i in range(len(hf[split]))]
N_OPEN = 100
sel = sorted(rng_open.choice(len(pool), size=N_OPEN, replace=False).tolist())

open_rows = []
for k, idx in enumerate(sel):
    split, i = pool[idx]
    x = hf[split][i]
    raw = x["image"]["bytes"]                      # оригінальні байти зображення, без перекодування
    ext = ".jpg" if raw[:3] == b"\xff\xd8\xff" else ".png"
    path = f"data/open/card_{k:03d}{ext}"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    open_rows.append({"k": k, "split": split, "index": i, "id": x["id"], "path": path,
                      "gt": x["transcription"].replace("\r\n", "\n"),
                      "reference": x["source_reference"]})
ods = pd.DataFrame(open_rows)
ods.to_csv("data/open/perestoroha_ids.csv", index=False, lineterminator="\n")
print("Карток:", len(ods), "| SHA-256 списку:", sha256_file("data/open/perestoroha_ids.csv"))
print(ods.split.value_counts().to_dict())
print("Середня довжина еталону, символів:", round(ods["gt"].str.len().mean(), 1))

# %% [markdown]
# **Історичні графеми.** Літери, яких немає в сучасному українському алфавіті: ѣ (ять), ъ (єр),
# ы (єри), ѳ (фіта), ѵ (іжиця), э. Літери є та і в карток трапляються, але вони є й у сучасній
# абетці, тому в окрему групу не входять.

# %%
HIST = set("ѣъыѳѵэѢЪЫѲѴЭ")
cnt_all = Counter("".join(ods["gt"]))
n_chars = sum(v for c, v in cnt_all.items() if not c.isspace())
hist_cnt = {c: cnt_all[c] for c in sorted(HIST) if cnt_all[c]}
print("Історичні графеми в еталоні 100 карток:", hist_cnt)
print(f"Частка історичних графем серед непробільних символів: {sum(hist_cnt.values()) / n_chars:.1%}")
print(f"Карток з хоча б однією історичною графемою: {ods['gt'].apply(lambda t: any(c in HIST for c in t)).mean():.0%}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, r in zip(axes.ravel(), ods.head(4).itertuples()):
    ax.imshow(cv2.cvtColor(imread_color(r.path), cv2.COLOR_BGR2RGB))
    ax.set_title(r.id, fontsize=8)
    ax.axis("off")
fig.suptitle(f"Приклади карток Perestoroha OCR\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
savefig(fig, "fig0b_open_samples")

# %% [markdown]
# ## 4. Гіпотези (записано до запуску розпізнавання)
#
# **Г1 (крива стійкості).** Жоден інструмент не перевищить поріг CER = 10 % на жодному рівні
# якості JPEG: літери висотою 22–35 px значно більші за блок стиснення 8×8 px, тому навіть при
# якості 5 % контури лишаються читабельними (рис. 0), з'являються лише «ореоли» навколо штрихів.
# На чистих зображеннях (якість ≥ 30 %) Tesseract буде точнішим за VLM (CER ≤ 2 % проти 2–4 %),
# бо VLM може змінювати пунктуацію, тире й апострофи. При якості 5 % CER Tesseract зросте до
# 3–6 %, а VLM майже не зміниться, тому криві перетнуться між якістю 10 % і 5 %.
#
# **Г2 (попередня обробка).** Для Tesseract при якості 10 % ланцюжок «збільшення ×2 → non-local
# means → Оцу» знизить CER приблизно на третину: NLM згладить блокові артефакти, а Оцу прибере
# сірі «ореоли». Саме лише збільшення не дасть покращення (збільшуються й артефакти), а
# адаптивна бінаризація буде гіршою за Оцу, бо на рівномірному білому тлі вона перетворює
# залишки артефактів на плями. Для VLM та сама обробка погіршить CER на 0,5–2 в. п., бо модель
# навчена на природних, а не бінаризованих зображеннях.
#
# **Г3 (відкритий набір).** На картках Perestoroha Tesseract матиме CER понад 80 %, а частина
# карток дасть повний провал (CER ≥ 1): він навчений на друкованих шрифтах, а не на стилізованому
# скорописі, і графем ѣ, ы, ъ немає в його алфавіті ukr. VLM матиме CER 25–40 %, тобто щонайменше
# вдвічі менше за Tesseract.
#
# **Г4 (особливість варіанта).** CER на позиціях історичних графем у VLM буде щонайменше вдвічі
# вищим, ніж на звичайних літерах. Зі звичайною інструкцією VLM «осучаснить» не менше 20 %
# історичних графем (ѣ → і/е, ы → и, кінцевий ъ → ∅), але для ѣ частіше траплятиметься заміна на
# схожу за формою літеру (ь), ніж на сучасний відповідник — як у пробному запуску («лѣто» →
# «льто»). Інструкція, яка явно вимагає зберегти дореформений правопис, зменшить частку
# «осучаснень» щонайменше вдвічі. Tesseract поверне 0 % правильних ѣ та ы (графеми поза алфавітом
# моделі).
#
# **Г5 (власні дані).** CER на власних фото рукопису буде щонайменше у 10 разів вищим, ніж на
# синтетичних рядках при якості 100 %, для обох інструментів (domain gap: рукопис, папір у
# клітинку, нерівне освітлення, перспектива): Tesseract — понад 70 %, VLM — 15–30 %. Найгірша
# умова для Tesseract — зйомка під кутом (рядки нахилені й мають різну висоту, а режим --psm 6
# очікує рівний блок тексту); для VLM — бічне світло з тінню, бо в затіненій частині слабкий
# контраст синьої ручки з папером.

# %% [markdown]
# ## 5. Попередня обробка
#
# Кроки: `upscale` — збільшення ×2 (кубічна інтерполяція); `nlm` — non-local means
# (`cv2.fastNlMeansDenoising`, h = 15), добре згладжує блокові артефакти JPEG, зберігаючи краї;
# `median` — медіанний фільтр 3×3; `otsu` — глобальна бінаризація Оцу; `adaptive` — адаптивна
# гаусова бінаризація (вікно 31, C = 15).

# %%
def preprocess(img, steps=()):
    out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if "upscale" in steps:
        out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if "nlm" in steps:
        out = cv2.fastNlMeansDenoising(out, None, h=15, templateWindowSize=7, searchWindowSize=21)
    if "median" in steps:
        out = cv2.medianBlur(out, 3)
    if "otsu" in steps:
        out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if "adaptive" in steps:
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
    return out


PREP = {"raw": (), "prep": ("upscale", "nlm", "otsu")}

demo = imread_gray(f"data/synth/000_{FACTOR}_10.png")
fig, axes = plt.subplots(4, 1, figsize=(8, 3.6))
for ax, (name, steps) in zip(axes, [("без обробки", ()), ("збільшення ×2", ("upscale",)),
                                     ("+ NLM", ("upscale", "nlm")), ("+ Оцу", ("upscale", "nlm", "otsu"))]):
    ax.imshow(preprocess(demo, steps), cmap="gray", vmin=0, vmax=255)
    ax.set_title(name, fontsize=8, loc="left")
    ax.axis("off")
fig.suptitle(f"Попередня обробка рядка при якості JPEG 10 %\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
savefig(fig, "fig0c_preprocessing")

# %% [markdown]
# ## 6. Інструменти розпізнавання
#
# * **Tesseract**: `--oem 1` (LSTM), `--psm 7` для рядка, `--psm 6` для картки/сторінки, мова `ukr+eng`.
#   Впевненість — середнє `conf` слів / 100.
# * **VLM Qwen2.5-VL 7B** (Ollama, Q4_K_M), `temperature = 0`, `seed = SEED`. Модель не повертає
#   впевненості, тому як її замінник використано **середню ймовірність токенів відповіді**
#   `exp(mean(logprob))`. Це не ймовірність правильності, а лише міра «невпевненості» декодера;
#   її придатність перевіряється так само, як для Tesseract (Спірмен, крива ризик-покриття).

# %%
def ocr_tesseract(img, lang=TESS_LANG, psm=7):
    t0 = time.perf_counter()
    d = pytesseract.image_to_data(img, lang=lang, config=f"--oem 1 --psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
    text = " ".join(w for w, _ in words)
    conf = np.mean([c for _, c in words]) / 100 if words else 0.0
    return text, float(conf), time.perf_counter() - t0


PROMPT_BASE = ("Перепиши дослівно весь текст із зображення, зберігаючи рядки. "
               "Нічого не виправляй, не перекладай і не додавай пояснень.")
PROMPT_HIST = ("Перепиши дослівно весь текст із зображення, зберігаючи рядки. Текст написаний "
               "староукраїнським правописом XVII століття: зберігай кожну літеру точно так, як "
               "вона написана, зокрема ѣ, ъ, ы, ѳ, ѵ, є, і. Не замінюй їх сучасними літерами, "
               "нічого не виправляй, не перекладай і не додавай пояснень.")


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def ocr_vlm(img, prompt=PROMPT_BASE, max_tokens=256, retries=3):
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = ollama.chat(model=VLM_MODEL, logprobs=True,
                            options={"temperature": 0, "seed": SEED, "num_predict": max_tokens},
                            messages=[{"role": "user", "content": prompt, "images": [png_bytes(img)]}])
            break
        except ollama.ResponseError as e:   # разові збої llama-server (CUDA) — повторюємо запит
            print("  повтор VLM:", str(e)[:80])
            time.sleep(5)
    else:
        raise RuntimeError("VLM недоступна")
    text = r.message.content.strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()   # прибираємо Markdown-огорожу
    lps = [t.logprob for t in (r.logprobs or [])]
    conf = float(np.exp(np.mean(lps))) if lps else 0.0
    return text, conf, time.perf_counter() - t0


ENGINES = {"Tesseract": lambda im: ocr_tesseract(im, psm=7),
           "VLM": lambda im: ocr_vlm(im, max_tokens=128)}

_ = ocr_vlm(render_line("Прогрів моделі", FONTS[0]), max_tokens=8)  # прогрів: модель завантажується в пам'ять
# Примітка: на ПОРОЖНЬОМУ білому зображенні 64×256 модель «прочитала» «Всім привіт! Як вам досить? ...» —
# приклад галюцинації VLM; для прогріву тому використовується рядок з текстом.
print("VLM готова")


def run_cached(name, make_rows):
    """Результати розпізнавання кешуються в results/<name>.csv, щоб ноутбук перезапускався швидко."""
    path = f"results/{name}.csv"
    if os.path.exists(path) and not FORCE:
        return pd.read_csv(path, keep_default_na=False)
    df = pd.DataFrame(make_rows())
    df.to_csv(path, index=False, lineterminator="\n")
    return pd.read_csv(path, keep_default_na=False)


def progress(k, n, t0, every=50):
    if k % every == 0 or k == n:
        print(f"  {k}/{n}, {time.perf_counter() - t0:.0f} с", flush=True)

# %% [markdown]
# ## 7. Нормалізація тексту та метрики
#
# Правила нормалізації (однакові для всіх інструментів):
# 1. Unicode NFC (й, ї як один символ).
# 2. Усі апострофи (’ ʼ ‘ ` ´) → `'`; «ялинки», „лапки“ та “лапки” → `"`.
# 3. Будь-які послідовності пробілів і переведень рядка → один пробіл; пробіли по краях прибираються.
# 4. Регістр і пунктуацію **не** ігноруємо: у варіанті досліджується точне відтворення
#    правопису, тож будь-яка зміна символу є помилкою. Історичні графеми **не** замінюються
#    сучасними (окремо рахується CER після «осучаснювальної» нормалізації — розділ 13).
#
# CER = (S + D + I) / N (символи), WER — те саме для слів. Усереднення **макро**: середнє CER
# окремих зразків (кожен рядок/картка має однакову вагу); мікро-CER наводиться окремо в таблиці.

# %%
APOSTROPHES = "\u2019\u02bc\u2018\u0060\u00b4"
QUOTES = "\u00ab\u00bb\u201e\u201c\u201d"


def normalize(s):
    s = unicodedata.normalize("NFC", str(s))
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    for q in QUOTES:
        s = s.replace(q, '"')
    return re.sub(r"\s+", " ", s).strip()


def cer(gt, pred):
    gt, pred = normalize(gt), normalize(pred)
    return Levenshtein.distance(gt, pred) / max(1, len(gt))


def wer(gt, pred):
    gt, pred = normalize(gt).split(), normalize(pred).split()
    return Levenshtein.distance(gt, pred) / max(1, len(gt))


def add_metrics(df):
    df = df.copy()
    df["pred"] = df["pred"].astype(str)
    df["cer"] = [cer(g, p) for g, p in zip(df["gt"], df.pred)]
    df["wer"] = [wer(g, p) for g, p in zip(df["gt"], df.pred)]
    df["len_ratio"] = [len(normalize(p)) / max(1, len(normalize(g))) for g, p in zip(df["gt"], df.pred)]
    df["dist"] = [Levenshtein.distance(normalize(g), normalize(p)) for g, p in zip(df["gt"], df.pred)]
    df["n"] = [len(normalize(g)) for g in df["gt"]]
    return df


def boot_ci(x, n_boot=2000, alpha=0.05):
    """Бутстреп-інтервал для середнього значення метрики."""
    x = np.asarray(x, float)
    means = rng_boot.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def paired_boot(a, b, n_boot=2000, alpha=0.05):
    """Бутстреп парної різниці середніх (a - b) на тих самих зразках."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng_boot.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return d.mean(), lo, hi, (means <= 0).mean()


print(cer("інженерія даних", "інженерія даиих"), wer("інженерія даних", "інженерія даиих"))

# %% [markdown]
# ## 8. Розпізнавання синтетичного набору
#
# 240 зображень × 2 інструменти × 2 варіанти обробки = 960 запусків. Час включає попередню обробку.

# %%
def synth_rows(frame=None):
    frame = ds if frame is None else frame
    rows, t0 = [], time.perf_counter()
    for k, r in enumerate(frame.itertuples(), 1):
        img = imread_gray(r.path)
        for prep_name, steps in PREP.items():
            tp = time.perf_counter()
            pim = preprocess(img, steps)
            tp = time.perf_counter() - tp
            for eng, fn in ENGINES.items():
                text, conf, sec = fn(pim)
                rows.append({"id": r.id, "level": r.level, "engine": eng, "prep": prep_name,
                             "gt": r.gt, "pred": text, "conf": conf, "sec": sec + tp})
        progress(k, len(frame), t0)
    return rows


res = add_metrics(run_cached("synth", synth_rows))
res["config"] = res.engine + " / " + res.prep

summary = (res.groupby(["engine", "prep", "level"])
           .agg(CER_mean=("cer", "mean"), CER_median=("cer", "median"), WER_mean=("wer", "mean"),
                empty=("pred", lambda s: (s.str.strip() == "").mean()), ms=("sec", "mean")))
summary["ms"] *= 1000
print(summary.round(3).to_string())

# %%
tab = res.pivot_table(index="config", columns="level", values="cer", aggfunc="mean")[LEVELS] * 100
print("Середній CER, %, залежно від якості JPEG")
print(tab.round(1).to_string())
micro = res.groupby(["config", "level"]).apply(lambda g: g.dist.sum() / g.n.sum()).unstack()[LEVELS] * 100
print("\nМікро-CER, %")
print(micro.round(1).to_string())
print("\nСередній WER, %")
print((res.pivot_table(index="config", columns="level", values="wer", aggfunc="mean")[LEVELS] * 100).round(1).to_string())

# %% [markdown]
# ## 9. Графіки метрик
# ### 9.1. Крива стійкості до стиснення JPEG

# %%
XPOS = {lvl: k for k, lvl in enumerate(LEVELS)}
fig, ax = plt.subplots(figsize=(8, 4.5))
ci_table = []
for cfg, g in res.groupby("config"):
    by_lvl = g.groupby("level")["cer"]
    mean = by_lvl.mean()[LEVELS]
    ci = np.array([boot_ci(by_lvl.get_group(l)) for l in LEVELS])
    xs = [XPOS[l] for l in LEVELS]
    ls = "-" if cfg.endswith("raw") else "--"
    ax.plot(xs, mean.values, marker="o", ls=ls, label=cfg)
    ax.fill_between(xs, ci[:, 0], ci[:, 1], alpha=0.15)
    for l, m, (lo, hi) in zip(LEVELS, mean.values, ci):
        ci_table.append({"config": cfg, "level": l, "mean": m, "lo": lo, "hi": hi, "width": hi - lo})
ax.axhline(0.10, ls="--", c="grey", lw=1)
ax.text(0, 0.11, "поріг CER = 10%", color="grey", fontsize=8)
ax.set_xticks(range(len(LEVELS)), [str(l) for l in LEVELS])
ax.set_xlabel("якість JPEG, % (праворуч — сильніше стиснення)")
ax.set_ylabel("CER (середнє, 95% ДІ)")
ax.legend()
stamp(ax, "Крива стійкості до стиснення JPEG")
savefig(fig, "fig1_robustness")

ci_table = pd.DataFrame(ci_table)
print(ci_table.pivot(index="config", columns="level", values="mean")[LEVELS].mul(100).round(1).to_string())

# %% [markdown]
# **Питання 1.** Рівень, на якому CER уперше перевищує 10 %, і перетин кривих.

# %%
for cfg, g in ci_table.groupby("config"):
    g = g.set_index("level").loc[LEVELS]
    over = [l for l in LEVELS if g.loc[l, "mean"] > 0.10]
    print(f"{cfg:20s}: CER > 10% уперше при якості {over[0] if over else '— (не перевищує)'} %")

raw = ci_table[ci_table.config.str.endswith("raw")].pivot(index="level", columns="config", values="mean").loc[LEVELS]
sign = np.sign(raw["Tesseract / raw"] - raw["VLM / raw"])
print("\nЗнак різниці (Tesseract − VLM), raw:", dict(zip(LEVELS, sign.astype(int))))
cross = [(LEVELS[k], LEVELS[k + 1]) for k in range(len(LEVELS) - 1) if sign.iloc[k] != sign.iloc[k + 1]]
print("Перетин кривих між рівнями:", cross if cross else "немає")

# %% [markdown]
# ### 9.2. Бутстреп парної різниці (питання 3)

# %%
paired = []
for lvl in LEVELS:
    a = res[(res.level == lvl) & (res.config == "Tesseract / raw")].sort_values("id").cer.values
    b = res[(res.level == lvl) & (res.config == "VLM / raw")].sort_values("id").cer.values
    m, lo, hi, p_le0 = paired_boot(a, b)
    paired.append({"level": lvl, "diff_T_minus_V": m, "lo": lo, "hi": hi, "zero_in_CI": lo <= 0 <= hi,
                   "share_T_not_worse": p_le0})
paired = pd.DataFrame(paired)
print(paired.round(4).to_string(index=False))
LVL_MAX = int(paired.loc[paired.diff_T_minus_V.abs().idxmax(), "level"])
print("Рівень з найбільшою різницею між інструментами:", LVL_MAX)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.errorbar(range(len(LEVELS)), paired.diff_T_minus_V, yerr=[paired.diff_T_minus_V - paired.lo, paired.hi - paired.diff_T_minus_V],
            fmt="o", capsize=4)
ax.axhline(0, c="grey", lw=1)
ax.set_xticks(range(len(LEVELS)), [str(l) for l in LEVELS])
ax.set_xlabel("якість JPEG, %")
ax.set_ylabel("CER(Tesseract) − CER(VLM), 95% ДІ")
stamp(ax, "Бутстреп парної різниці CER (без обробки)")
savefig(fig, "fig1b_paired_bootstrap")

# %% [markdown]
# ### 9.3. Розподіл CER: діаграма розмаху і гістограма (питання 4)

# %%
LVL_BOX = LVL_MAX
sub = res[(res.level == LVL_BOX) & (res.prep == "raw")]
names = sorted(sub.engine.unique())
groups = [sub[sub.engine == e]["cer"].values for e in names]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
ax.boxplot(groups, tick_labels=names, showmeans=True)
for k, g in enumerate(groups, start=1):
    ax.scatter(np.full(len(g), k) + rng_plot.uniform(-0.08, 0.08, len(g)), g, s=12, alpha=0.5)
ax.set_ylabel("CER окремого рядка")
stamp(ax, f"Розподіл CER при якості JPEG {LVL_BOX} % (без обробки)")
ax = axes[1]
bins = np.linspace(0, max(1.0, sub.cer.max()), 21)
for e, g in zip(names, groups):
    ax.hist(g, bins=bins, alpha=0.5, label=e)
ax.set_xlabel("CER окремого рядка")
ax.set_ylabel("кількість рядків")
ax.legend()
stamp(ax, f"Гістограма CER, якість {LVL_BOX} %")
savefig(fig, "fig2_boxplot")

for e, g in zip(names, groups):
    q1, med, q3 = np.quantile(g, [0.25, 0.5, 0.75])
    print(f"{e:10s} середнє={g.mean():.3f} медіана={med:.3f} IQR=[{q1:.3f}; {q3:.3f}] max={g.max():.3f} "
          f"CER>=1: {(g >= 1).mean():.0%} порожніх: {(sub[sub.engine == e].pred.str.strip() == '').mean():.0%}")
fails = res[(res.cer >= 1) & (res.prep == "raw")]
print("\nПовні провали (CER ≥ 1) по всіх рівнях, raw:")
print(fails.groupby(["engine", "level"]).size().to_string() if len(fails) else "немає")

# %% [markdown]
# ### 9.4. Впевненість проти помилки і крива ризик-покриття (питання 5)

# %%
def coverage_at(g, max_cer=0.02):
    """Максимальна частка автоматично прийнятих зразків, для якої середній CER прийнятих ≤ max_cer."""
    g = g.sort_values("conf", ascending=False)
    risk = g.cer.cumsum().values / np.arange(1, len(g) + 1)
    ok = np.where(risk <= max_cer)[0]
    if len(ok) == 0:
        return 0.0, None, risk
    k = ok.max()
    return (k + 1) / len(g), g.conf.values[k], risk


fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for eng, g in res[res.prep == "raw"].groupby("engine"):
    rho, p = spearmanr(g.conf, g.cer)
    axes[0].scatter(g.conf, g.cer, s=10, alpha=0.5, label=f"{eng}: ρ = {rho:.2f}")
    cov, thr, risk = coverage_at(g)
    coverage = np.arange(1, len(g) + 1) / len(g)
    axes[1].plot(coverage, risk, label=eng)
    print(f"{eng:10s} Спірмен ρ = {rho:.3f} (p = {p:.1e}); CER ≤ 2%: покриття {cov:.0%}, "
          f"поріг впевненості {thr if thr is None else round(thr, 3)}")
axes[0].set_xlabel("впевненість інструмента")
axes[0].set_ylabel("CER")
axes[0].legend()
stamp(axes[0], "Впевненість проти помилки (синтетика, raw)")
axes[1].axhline(0.02, ls="--", c="grey", lw=1)
axes[1].text(0.02, 0.022, "CER = 2%", color="grey", fontsize=8)
axes[1].set_xlabel("частка автоматично прийнятих рядків (coverage)")
axes[1].set_ylabel("середній CER прийнятих рядків")
axes[1].legend()
stamp(axes[1], "Крива ризик-покриття")
savefig(fig, "fig3_confidence")

# %% [markdown]
# ### 9.5. Компроміс якість-швидкість (питання 6)

# %%
agg = res.groupby("config").agg(cer=("cer", "mean"), ms=("sec", "mean")).reset_index()
agg["ms"] *= 1000
agg["dominated_by"] = [", ".join(o.config for o in agg.itertuples()
                                 if o.config != r.config and o.cer <= r.cer and o.ms <= r.ms
                                 and (o.cer < r.cer or o.ms < r.ms)) for r in agg.itertuples()]
fig, ax = plt.subplots(figsize=(7, 4.4))
for r in agg.itertuples():
    ax.scatter(r.ms, r.cer, s=70, marker="o" if not r.dominated_by else "x")
    ax.annotate(f"{r.config}\n{r.ms:.0f} мс, CER {r.cer:.3f}", (r.ms, r.cer), textcoords="offset points",
                xytext=(6, 4), fontsize=8)
front = agg[agg.dominated_by == ""].sort_values("ms")
ax.plot(front.ms, front.cer, c="grey", lw=1, ls=":", label="фронт Парето")
ax.set_xscale("log")
ax.set_xlabel("середній час на рядок, мс (лог. шкала)")
ax.set_ylabel("середній CER (усі рівні JPEG)")
ax.legend()
stamp(ax, "Компроміс якість-швидкість")
savefig(fig, "fig4_pareto")
print(agg.round(4).to_string(index=False))

# %% [markdown]
# ### 9.6. Абляційний експеримент (питання 7)
#
# Tesseract — усі 6 конфігурацій на всіх рівнях (запуски дешеві); VLM — 4 конфігурації на рівні
# `ABL_LEVEL`. Кроки додаються послідовно, останні два рядки — альтернативи окремим крокам.

# %%
ABL_LEVEL = 10
ablation = {"без обробки": (),
            "+ збільшення ×2": ("upscale",),
            "+ NLM-фільтр": ("upscale", "nlm"),
            "+ бінаризація Оцу": ("upscale", "nlm", "otsu"),
            "адаптивна замість Оцу": ("upscale", "nlm", "adaptive"),
            "медіанний замість NLM": ("upscale", "median", "otsu")}
ABL_VLM = list(ablation)[:4]


def ablation_rows():
    rows, t0 = [], time.perf_counter()
    for k, r in enumerate(ds.itertuples(), 1):
        img = imread_gray(r.path)
        for name, steps in ablation.items():
            pim = preprocess(img, steps)
            text, conf, sec = ocr_tesseract(pim)
            rows.append({"engine": "Tesseract", "level": r.level, "id": r.id, "step": name, "gt": r.gt, "pred": text})
            if r.level == ABL_LEVEL and name in ABL_VLM:
                text, conf, sec = ocr_vlm(pim, max_tokens=128)
                rows.append({"engine": "VLM", "level": r.level, "id": r.id, "step": name, "gt": r.gt, "pred": text})
        progress(k, len(ds), t0)
    return rows


abl_all = add_metrics(run_cached("ablation", ablation_rows))
abl_tab = abl_all[abl_all.engine == "Tesseract"].pivot_table(index="step", columns="level", values="cer", sort=False)[LEVELS]
print("Tesseract: середній CER, % за кроками обробки і рівнями JPEG")
print((abl_tab * 100).round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, eng in zip(axes, ["Tesseract", "VLM"]):
    a = abl_all[(abl_all.engine == eng) & (abl_all.level == ABL_LEVEL)].groupby("step", sort=False)["cer"].mean()
    ax.barh(a.index[::-1], a.values[::-1], color="C0" if eng == "Tesseract" else "C1")
    for y, v in enumerate(a.values[::-1]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel(f"середній CER ({eng})")
    stamp(ax, f"Внесок кроків обробки, {eng}, якість {ABL_LEVEL} %")
savefig(fig, "fig5_ablation")
print(abl_all[abl_all.level == ABL_LEVEL].pivot_table(index="step", columns="engine", values="cer", sort=False).round(4).to_string())

# %% [markdown]
# ### 9.7. Аналіз помилок: найчастіші заміни символів (питання 8)

# %%
def sym(c):
    return "␣" if c == " " else c


def confusions(gt, pred):
    gt, pred = normalize(gt), normalize(pred)
    pairs = []
    for op in Levenshtein.editops(gt, pred):
        if op.tag == "replace":
            pairs.append(f"{sym(gt[op.src_pos])} → {sym(pred[op.dest_pos])}")
        elif op.tag == "delete":
            pairs.append(f"{sym(gt[op.src_pos])} → ∅")
        else:
            pairs.append(f"∅ → {sym(pred[op.dest_pos])}")
    return pairs


def plot_confusions(frame, title, fname, top_n=15):
    engines = sorted(frame.engine.unique())
    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 4.8))
    axes = np.atleast_1d(axes)
    tops = {}
    for ax, eng in zip(axes, engines):
        part = frame[(frame.engine == eng) & (frame.cer < 0.5)]
        c = Counter()
        for r in part.itertuples():
            c.update(confusions(r.gt, r.pred))
        top = pd.Series(dict(c.most_common(top_n)))
        tops[eng] = top
        if len(top):
            ax.barh(top.index[::-1], top.values[::-1])
        ax.set_xlabel("кількість")
        stamp(ax, f"{title}: {eng} (рядків з CER<0,5: {len(part)})")
    savefig(fig, fname)
    return tops


tops_synth = plot_confusions(res[res.prep == "raw"], "Заміни (синтетика)", "fig6_confusions")
for eng, top in tops_synth.items():
    print(eng, top.head(8).to_dict())

# %% [markdown]
# ## 10. Розпізнавання відкритого набору Perestoroha
#
# Tesseract у режимі `--psm 6` (блок тексту) без обробки і з адаптивною бінаризацією; VLM з двома
# інструкціями: базовою (`base`) і з вимогою зберегти дореформений правопис (`hist`).

# %%
OPEN_CFG = {
    ("Tesseract", "raw"): lambda p: ocr_tesseract(imread_gray(p), psm=6),
    ("Tesseract", "adaptive"): lambda p: ocr_tesseract(preprocess(imread_gray(p), ("adaptive",)), psm=6),
    ("VLM", "base"): lambda p: ocr_vlm(imread_color(p), PROMPT_BASE, 512),
    ("VLM", "hist"): lambda p: ocr_vlm(imread_color(p), PROMPT_HIST, 512),
}


def open_rows_fn():
    rows, t0 = [], time.perf_counter()
    for k, r in enumerate(ods.itertuples(), 1):
        for (eng, prep_name), fn in OPEN_CFG.items():
            text, conf, sec = fn(r.path)
            rows.append({"k": r.k, "id": r.id, "engine": eng, "prep": prep_name, "gt": r.gt,
                         "pred": text, "conf": conf, "sec": sec})
        progress(k, len(ods), t0, every=10)
    return rows


ores = add_metrics(run_cached("open", open_rows_fn))
ores["config"] = ores.engine + " / " + ores.prep
otab = ores.groupby("config").agg(CER_mean=("cer", "mean"), CER_median=("cer", "median"),
                                  WER_mean=("wer", "mean"), len_ratio=("len_ratio", "median"),
                                  fails=("cer", lambda s: (s >= 1).mean()), s=("sec", "mean"))
otab["CER_micro"] = ores.groupby("config").apply(lambda g: g.dist.sum() / g.n.sum())
otab[["CI_lo", "CI_hi"]] = [boot_ci(g.cer) for _, g in ores.groupby("config")]
print(otab.round(3).to_string())

m, lo, hi, _ = paired_boot(ores[ores.config == "Tesseract / raw"].sort_values("k").cer,
                           ores[ores.config == "VLM / base"].sort_values("k").cer)
print(f"\nПарна різниця CER Tesseract/raw − VLM/base: {m:.3f} [{lo:.3f}; {hi:.3f}]")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cfgs = sorted(ores.config.unique())
gs = [ores[ores.config == c].cer.values for c in cfgs]
axes[0].boxplot(gs, tick_labels=[c.replace(" / ", "\n") for c in cfgs], showmeans=True)
for k, g in enumerate(gs, start=1):
    axes[0].scatter(np.full(len(g), k) + rng_plot.uniform(-0.08, 0.08, len(g)), g, s=10, alpha=0.5)
axes[0].set_ylabel("CER картки")
stamp(axes[0], "Perestoroha: розподіл CER (100 карток)")
for eng, g in ores[ores.prep.isin(["raw", "base"])].groupby("engine"):
    rho, _ = spearmanr(g.conf, g.cer)
    cov, thr, risk = coverage_at(g, 0.10)
    axes[1].plot(np.arange(1, len(g) + 1) / len(g), risk, label=f"{eng}: ρ = {rho:.2f}")
    print(f"{eng}: ρ = {rho:.3f}; CER ≤ 10%: покриття {cov:.0%}, поріг {thr}")
axes[1].axhline(0.10, ls="--", c="grey", lw=1)
axes[1].set_xlabel("частка автоматично прийнятих карток")
axes[1].set_ylabel("середній CER прийнятих")
axes[1].legend()
stamp(axes[1], "Perestoroha: крива ризик-покриття")
savefig(fig, "fig7_open")

tops_open = plot_confusions(ores[ores.prep.isin(["raw", "base"])], "Заміни (Perestoroha)", "fig6b_confusions_open")
for eng, top in tops_open.items():
    print(eng, top.head(10).to_dict())

# %% [markdown]
# ## 11. Власні дані
#
# Фото в `my_photos/` (імена латиницею), поруч `.txt` з ручною транскрипцією:
# `page01_flat|shadow|angle.jpg` — рукописна сторінка з «Інститутки» в трьох умовах зйомки;
# `old01.jpg` — власноруч написаний текст з дореформеними літерами.
#
# Кожне фото: EXIF-орієнтація → відтінки сірого → довша сторона 2000 px. Конфігурації:
# `raw`; `warp` — пошук аркуша (Кенні + найбільший чотирикутник) і корекція перспективи;
# `warp+adaptive` — плюс адаптивна бінаризація. VLM отримує кольорове зображення (довша сторона
# 1600 px) без обробки та після корекції перспективи.

# %%
def exif_info(path):
    """Дата зйомки та модель камери з EXIF (доказ, що фото зроблене вами)."""
    exif = Image.open(path).getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    sub_ifd = exif.get_ifd(0x8769)
    taken = sub_ifd.get(36867) or tags.get("DateTime")
    return taken, tags.get("Model"), tags.get("Make")


def load_photo(path, long_side):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    k = long_side / max(im.size)
    im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, dif = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[dif.argmin()], pts[s.argmax()], pts[dif.argmax()]])


def find_page(gray, min_area=0.25):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 30, 90), np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area * gray.size:
            return order_corners(approx)
    return None


def warp_page(img, corners, width=1400):
    tl, tr, br, bl = corners
    h = int(width * max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            / max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst = np.float32([[0, 0], [width, 0], [width, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (width, h))


photos = sorted(f"my_photos/{f}" for f in os.listdir("my_photos")
                if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg"))
print("Фото:", [os.path.basename(p) for p in photos])


def transcription(photo_name):
    """Ручна транскрипція: transcriptions/<ім'я>.txt (або поруч із фото)."""
    stem = os.path.splitext(os.path.basename(photo_name))[0]
    for path in (f"transcriptions/{stem}.txt", f"my_photos/{stem}.txt"):
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                return f.read()
    raise FileNotFoundError(f"немає транскрипції для {photo_name}")


def own_rows_fn():
    rows = []
    for p in photos:
        gt = transcription(p)
        taken, camera, make = exif_info(p)
        big = load_photo(p, 2000)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        corners = find_page(gray)
        variants = {"raw": (gray, big)}
        if corners is not None:
            variants["warp"] = (warp_page(gray, corners), warp_page(big, corners))
        base = os.path.basename(p)
        for prep_name, (g_img, c_img) in variants.items():
            imwrite(f"tmp/{os.path.splitext(base)[0]}_{prep_name}.png", g_img)
            for tess_prep, im in [(prep_name, g_img), (prep_name + "+adaptive", preprocess(g_img, ("adaptive",)))]:
                text, conf, sec = ocr_tesseract(im, psm=6)
                rows.append({"file": base, "engine": "Tesseract", "prep": tess_prep, "gt": gt, "pred": text,
                             "conf": conf, "sec": sec, "taken": taken, "camera": camera, "make": make,
                             "page_found": corners is not None})
            vimg = c_img
            k = 1600 / max(vimg.shape[:2])
            vimg = cv2.resize(vimg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            prompts = {"base": PROMPT_BASE, "hist": PROMPT_HIST} if base.startswith("old") else {"base": PROMPT_BASE}
            for pname, prompt in prompts.items():
                text, conf, sec = ocr_vlm(vimg, prompt, 1024)
                rows.append({"file": base, "engine": "VLM", "prep": f"{prep_name}/{pname}", "gt": gt, "pred": text,
                             "conf": conf, "sec": sec, "taken": taken, "camera": camera, "make": make,
                             "page_found": corners is not None})
        print(" ", base, "готово", flush=True)
    return rows


if photos:
    own = add_metrics(run_cached("own", own_rows_fn))
    print(own[["file", "taken", "camera", "page_found"]].drop_duplicates("file").to_string(index=False))
    print(own.pivot_table(index="file", columns=["engine", "prep"], values="cer").round(3).T.to_string())
else:
    own = pd.DataFrame()
    print("Каталог my_photos порожній — додайте фото і транскрипції")

# %%
if len(own):
    files = list(dict.fromkeys(own.file))
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4.5))
    for ax, f in zip(np.atleast_1d(axes), files):
        ax.imshow(cv2.cvtColor(load_photo(f"my_photos/{f}", 1000), cv2.COLOR_BGR2RGB))
        g = own[own.file == f]
        ax.set_title(f"{f}\nT raw CER={g[(g.engine == 'Tesseract') & (g.prep == 'raw')].cer.iloc[0]:.2f}, "
                     f"V CER={g[(g.engine == 'VLM') & (g.prep == 'raw/base')].cer.iloc[0]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Власні фото\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
    savefig(fig, "fig8_own_photos")

# %% [markdown]
# ### 11.1. Порівняння трьох груп даних (питання 9)

# %%
grp = []


def add_group(name, frame, cfg_map):
    for eng, cond in cfg_map.items():
        g = frame[cond(frame) & (frame.engine == eng)]
        if len(g):
            lo, hi = boot_ci(g.cer) if len(g) > 1 else (np.nan, np.nan)
            grp.append({"group": name, "engine": eng, "n": len(g), "cer": g.cer.mean(), "lo": lo, "hi": hi,
                        "wer": g.wer.mean()})


add_group("синтетика, JPEG 100%", res, {e: (lambda f: (f.level == 100) & (f.prep == "raw")) for e in ["Tesseract", "VLM"]})
add_group("синтетика, усі рівні", res, {e: (lambda f: f.prep == "raw") for e in ["Tesseract", "VLM"]})
add_group("Perestoroha (100 карток)", ores, {"Tesseract": lambda f: f.prep == "raw", "VLM": lambda f: f.prep == "base"})
if len(own):
    add_group("власні: сторінка «Інститутки»", own[own.file.str.startswith("page")],
              {"Tesseract": lambda f: f.prep == "raw", "VLM": lambda f: f.prep == "raw/base"})
    add_group("власні: дореформений текст", own[own.file.str.startswith("old")],
              {"Tesseract": lambda f: f.prep == "raw", "VLM": lambda f: f.prep == "raw/base"})
grp = pd.DataFrame(grp)
print(grp.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.5))
gnames = list(dict.fromkeys(grp.group))
for j, eng in enumerate(["Tesseract", "VLM"]):
    g = grp[grp.engine == eng].set_index("group").reindex(gnames)
    x = np.arange(len(gnames)) + (j - 0.5) * 0.38
    ax.bar(x, g.cer, width=0.38, label=eng,
           yerr=[(g.cer - g.lo).fillna(0), (g.hi - g.cer).fillna(0)], capsize=3)
    for xi, v in zip(x, g.cer):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(gnames)), [textwrap.fill(n, 18) for n in gnames], fontsize=8)
ax.set_ylabel("середній CER (95% ДІ)")
ax.legend()
stamp(ax, "CER на синтетичних, відкритих і власних даних")
savefig(fig, "fig9_groups")

if len(own):
    pg = own[own.file.str.startswith("page") & own.prep.isin(["raw", "raw/base"])]
    print(pg.pivot_table(index="file", columns="engine", values="cer").round(3).to_string())

# %% [markdown]
# ## 12. Особливість варіанта: чи «осучаснює» VLM правопис
#
# Для кожної позиції еталону з історичною графемою за вирівнюванням Левенштейна визначаємо, що
# повернув інструмент: ту саму графему (правильно), її сучасний відповідник (**осучаснення**:
# ѣ → і/е/ї, ы → и, ъ → ∅/ь, ѳ → ф, ѵ → і/и/в, э → е), інший символ або нічого.
# **CER історичних графем** = частка позицій історичних графем, відтворених неправильно;
# так само рахується CER решти (звичайних) символів.

# %%
MODERN = {"ѣ": set("іеї"), "ы": set("и"), "ъ": {"", "ь"}, "ѳ": set("ф"), "ѵ": set("іив"), "э": set("е"),
          "Ѣ": set("ІЕЇ"), "Ы": set("И"), "Ъ": {"", "Ь"}, "Ѳ": set("Ф"), "Ѵ": set("ІИВ"), "Э": set("Е")}


def align_chars(gt, pred):
    """Список (символ еталону, символ результату або '' для видалення) для кожної позиції еталону."""
    out = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gt, pred):
        if tag == "equal" or tag == "replace":
            for k in range(i2 - i1):
                out.append((gt[i1 + k], pred[j1 + k] if j1 + k < j2 else ""))
        elif tag == "delete":
            out.extend((gt[i], "") for i in range(i1, i2))
    return out


def hist_stats(gt, pred):
    gt, pred = normalize(gt), normalize(pred)
    st = Counter()
    pairs = Counter()
    for g, p in align_chars(gt, pred):
        if g in HIST:
            st["n_hist"] += 1
            if p == g:
                st["ok_hist"] += 1
            elif p in MODERN[g]:
                st["modern"] += 1
                pairs[f"{g} → {p or '∅'}"] += 1
            else:
                st["other_hist"] += 1
                pairs[f"{g} → {p or '∅'}"] += 1
        elif not g.isspace():
            st["n_reg"] += 1
            st["ok_reg"] += p == g
    st["pred_hist"] = sum(c in HIST for c in pred)
    return st, pairs


def hist_table(frame):
    out, all_pairs = [], {}
    for cfg, g in frame.groupby("config"):
        tot, prs = Counter(), Counter()
        for r in g.itertuples():
            s, p = hist_stats(r.gt, r.pred)
            tot.update(s)
            prs.update(p)
        out.append({"config": cfg, "n_hist": tot["n_hist"],
                    "CER_hist": 1 - tot["ok_hist"] / max(1, tot["n_hist"]),
                    "CER_regular": 1 - tot["ok_reg"] / max(1, tot["n_reg"]),
                    "modernized": tot["modern"] / max(1, tot["n_hist"]),
                    "other_err_hist": tot["other_hist"] / max(1, tot["n_hist"]),
                    "hist_in_pred": tot["pred_hist"]})
        all_pairs[cfg] = prs
    return pd.DataFrame(out), all_pairs


htab, hpairs = hist_table(ores)
print("Perestoroha, 100 карток:")
print(htab.round(3).to_string(index=False))
for cfg, prs in hpairs.items():
    print(cfg, dict(prs.most_common(8)))

# %% [markdown]
# **CER після «осучаснювальної» нормалізації.** Якщо і еталон, і результат перевести в сучасні
# відповідники (ѣ → і, ы → и, кінцевий ъ → ∅, ѳ → ф, ѵ → і, э → е), різниця CER до і після показує,
# яку частку помилок спричинив саме правопис.

# %%
MOD_MAP = str.maketrans({"ѣ": "і", "ы": "и", "ѳ": "ф", "ѵ": "і", "э": "е", "Ѣ": "І", "Ы": "И", "Ѳ": "Ф", "Ѵ": "І", "Э": "Е"})


def modernize(s):
    s = normalize(s).translate(MOD_MAP)
    s = re.sub(r"[ъЪ](?=\W|$)", "", s)   # кінцевий єр
    return s.replace("ъ", "'")             # єр всередині слова як роздільник


ores["cer_mod"] = [cer(modernize(g), modernize(p)) for g, p in zip(ores["gt"], ores.pred)]
print(ores.groupby("config")[["cer", "cer_mod"]].mean().round(4).assign(
    share_due_to_orthography=lambda d: 1 - d.cer_mod / d.cer).round(3).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
x = np.arange(len(htab))
ax.bar(x - 0.2, htab.CER_regular, 0.4, label="звичайні символи")
ax.bar(x + 0.2, htab.CER_hist, 0.4, label="історичні графеми (ѣ ъ ы ѳ ѵ э)")
ax.bar(x + 0.2, htab.modernized, 0.4, fill=False, hatch="//", edgecolor="k", label="з них осучаснено")
for xi, a, b, m in zip(x, htab.CER_regular, htab.CER_hist, htab.modernized):
    ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(xi + 0.2, b + 0.01, f"{b:.2f}\n(осуч. {m:.2f})", ha="center", fontsize=7)
ax.set_xticks(x, [c.replace(" / ", "\n") for c in htab.config], fontsize=8)
ax.set_ylabel("частка неправильно відтворених позицій")
ax.legend(fontsize=8)
stamp(ax, "Perestoroha: CER історичних графем і звичайних символів")
ax = axes[1]
vlm_pairs = hpairs.get("VLM / base", Counter()) + hpairs.get("VLM / hist", Counter())
top_keys = [k for k, _ in vlm_pairs.most_common(12)]
yy = np.arange(len(top_keys))
for j, cfg in enumerate(["VLM / base", "VLM / hist"]):
    if cfg in hpairs:
        ax.barh(yy + (j - 0.5) * 0.4, [hpairs[cfg].get(k, 0) for k in top_keys], 0.4, label=cfg)
ax.set_yticks(yy, top_keys)
ax.invert_yaxis()
ax.set_xlabel("кількість")
ax.legend()
stamp(ax, "Що VLM повертає замість історичних графем")
savefig(fig, "fig10_historical")

if len(own) and own.file.str.startswith("old").any():
    oo = own[own.file.str.startswith("old")].copy()
    oo["config"] = oo.engine + " / " + oo.prep
    htab_own, hpairs_own = hist_table(oo)
    print("Власний дореформений текст:")
    print(htab_own.round(3).to_string(index=False))
    for cfg, prs in hpairs_own.items():
        print(cfg, dict(prs.most_common(8)))

# %% [markdown]
# ## 13. Питання 2: ширина довірчих смуг і збільшена вибірка
#
# Ширина бутстреп-інтервалу середнього зменшується приблизно як 1/√n, тому для звуження вдвічі
# потрібно вчетверо більше зразків. Перевіряємо: у найгіршій точці додаємо 120 нових рядків
# (разом 160) і порівнюємо ширину інтервалу.

# %%
worst = ci_table[ci_table.config.str.endswith("raw")].sort_values("width", ascending=False).iloc[0]
Q2_LEVEL = int(worst.level)
print(ci_table[ci_table.config.str.endswith("raw")].pivot(index="config", columns="level", values="width")[LEVELS].round(4).to_string())
print(f"Найширший інтервал: {worst.config}, якість {Q2_LEVEL} %, ширина {worst.width:.4f}")

extra_pool = [s for s in corpus if s not in set(lines)]
extra_lines = [str(s) for s in rng_extra.choice(extra_pool, size=3 * N_LINES, replace=False)]
ds_extra = build_synth(extra_lines, "x", rng_extra, [Q2_LEVEL])
ds_extra["id"] += 1000
PREP_SAVE = PREP
PREP = {"raw": ()}
res_extra = add_metrics(run_cached(f"q2_extra_{Q2_LEVEL}", lambda: synth_rows(ds_extra)))
PREP = PREP_SAVE
q2 = []
for eng in ["Tesseract", "VLM"]:
    a = res[(res.level == Q2_LEVEL) & (res.config == f"{eng} / raw")].cer.values
    b = np.concatenate([a, res_extra[res_extra.engine == eng].cer.values])
    for n, x in [(len(a), a), (len(b), b)]:
        lo, hi = boot_ci(x)
        q2.append({"engine": eng, "n": n, "mean": x.mean(), "lo": lo, "hi": hi, "width": hi - lo})
q2 = pd.DataFrame(q2)
q2["ratio_to_n40"] = q2.width / q2.groupby("engine").width.transform("first")
print(q2.round(4).to_string(index=False))

# %% [markdown]
# ## 14. П'ять найгірших зразків
#
# Беремо результати без обробки з усіх трьох груп; категорію причини визначаємо евристикою
# (порожній результат → детекція; довжина < 0,5 еталону → пропуск рядків/сегментація; довжина
# > 1,5 еталону → галюцинація; інакше — заміна символів) і перевіряємо вручну.

# %%
def category(r):
    if not str(r.pred).strip():
        return "детекція (порожній результат)"
    if r.len_ratio > 1.5:
        return "галюцинація / зайвий текст"
    if r.len_ratio < 0.5:
        return "сегментація / пропуск рядків"
    return "заміна символів"


pool_w = pd.concat([
    res[res.prep == "raw"].assign(group="синтетика", path=lambda d: [f"data/synth/{i:03d}_{FACTOR}_{l}.png" for i, l in zip(d.id, d.level)]),
    ores[ores.prep.isin(["raw", "base"])].assign(group="Perestoroha", path=lambda d: d.k.map(ods.set_index("k").path)),
] + ([own[own.prep.isin(["raw", "raw/base"])].assign(group="власні", path=lambda d: "my_photos/" + d.file)] if len(own) else []),
    ignore_index=True)
worst5 = pool_w.sort_values("cer", ascending=False).drop_duplicates("path").head(5).copy()
worst5["category"] = [category(r) for r in worst5.itertuples()]
print(worst5[["group", "engine", "path", "cer", "len_ratio", "category"]].to_string(index=False))
worst_vlm = pool_w[pool_w.engine == "VLM"].sort_values("cer", ascending=False).head(5).copy()
worst_vlm["category"] = [category(r) for r in worst_vlm.itertuples()]
print("\nП'ять найгірших результатів VLM:")
print(worst_vlm[["group", "path", "cer", "len_ratio", "category"]].to_string(index=False))


def aligned_cols(gt, pred):
    cols = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(normalize(gt), normalize(pred)):
        g, p = normalize(gt)[i1:i2], normalize(pred)[j1:j2]
        if tag == "equal":
            cols += [(c, c, "eq") for c in g]
        elif tag == "replace":
            L = max(len(g), len(p))
            for k in range(L):
                cols.append((g[k] if k < len(g) else " ", p[k] if k < len(p) else "·",
                             "sub" if k < len(g) and k < len(p) else ("del" if k < len(g) else "ins")))
        elif tag == "delete":
            cols += [(c, "·", "del") for c in g]
        else:
            cols += [(" ", c, "ins") for c in p]
    return cols


def draw_alignment(ax, gt, pred, width=70, max_rows=4):
    cols = aligned_cols(gt, pred)
    chunks = [cols[i:i + width] for i in range(0, len(cols), width)][:max_rows]
    ax.set_xlim(0, width)
    ax.set_ylim(len(chunks) * 2.4, 0)
    ax.axis("off")
    fw = ax.get_window_extent().width / width * 72 / ax.figure.dpi / 0.62
    color = {"sub": "#f8b4c8", "del": "#f8b4c8", "ins": "#fde68a"}
    for r, ch in enumerate(chunks):
        for x, (g, p, t) in enumerate(ch):
            for dy, c in [(0.6, g), (1.6, p)]:
                ax.text(x + 0.5, r * 2.4 + dy, c, family="DejaVu Sans Mono", fontsize=fw, ha="center", va="center",
                        bbox=dict(facecolor=color[t], edgecolor="none", pad=0.5) if t != "eq" else None)


def plot_worst(frame, fname, title):
    fig, axes = plt.subplots(len(frame), 2, figsize=(15, 2.6 * len(frame)), gridspec_kw={"width_ratios": [1, 2.2]})
    for (ax_i, ax_t), r in zip(np.atleast_2d(axes), frame.itertuples()):
        img = load_photo(r.path, 900) if r.group == "власні" else imread_color(r.path)
        ax_i.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_i.set_title(f"{r.group}, {r.engine}: CER={r.cer:.2f}, {r.category}", fontsize=8, loc="left")
        ax_i.axis("off")
        draw_alignment(ax_t, r.gt, r.pred)
    fig.suptitle(f"{title} (верхній рядок — еталон, нижній — результат; рожеве — заміна/видалення, "
                 f"жовте — вставлення)\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
    fig.tight_layout()
    savefig(fig, fname)


plot_worst(worst5, "fig11_worst5", "П'ять найгірших зразків")
plot_worst(worst_vlm, "fig11b_worst5_vlm", "П'ять найгірших результатів VLM")
for r in worst5.itertuples():
    print(f"--- {r.path} [{r.engine}] CER={r.cer:.3f}\nЕталон:    {normalize(r.gt)[:300]}\nРезультат: {normalize(r.pred)[:300]}")

# %% [markdown]
# ## 15. Постобробка (питання 8)
#
# Правила для найчастіших систематичних помилок; вимірюємо CER до і після на синтетиці (raw) і на
# картках. Правила підібрано за діаграмами замін (рис. 6, 6b).

# %%
LAT2CYR = str.maketrans("aceopxyiABCEHKMOPTXI", "асеорхуіАВСЕНКМОРТХІ")


def pp_homoglyphs(s):
    """Латинські літери всередині кириличних слів → кириличні відповідники."""
    return re.sub(r"\w+", lambda m: m.group(0).translate(LAT2CYR)
                  if re.search(r"[а-яіїєґѣъыѳѵ]", m.group(0), re.I) else m.group(0), s)


def pp_strip_edges(s):
    """Прибрати шумові символи на краях рядка: лапки, риски, вертикальні риски."""
    return re.sub(r"^[\"'|‘“\[\]\s]+|[|\[\]\s]+$", "", s)


def pp_quotes(s):
    """Лапки «ялинки»/„“ VLM → прямі (у нормалізації вони вже уніфіковані) і прибирання пробілу перед розділовими."""
    return re.sub(r"\s+([,.;:!?])", r"\1", s)


RULES = {"гомогліфи лат→кир": pp_homoglyphs, "шум на краях": pp_strip_edges, "пробіл перед розділовим": pp_quotes}


def eval_rules(frame, label):
    out = []
    base = frame.cer.mean()
    for eng, g in frame.groupby("engine"):
        row = {"data": label, "engine": eng, "CER_before": g.cer.mean()}
        acc = g.pred.astype(str)
        for name, fn in RULES.items():
            alone = np.mean([cer(gt, fn(p)) for gt, p in zip(g["gt"], g.pred.astype(str))])
            row[f"тільки «{name}»"] = alone
            acc = acc.map(fn)
        row["усі правила"] = np.mean([cer(gt, p) for gt, p in zip(g["gt"], acc)])
        out.append(row)
    return out


pp_tab = pd.DataFrame(eval_rules(res[res.prep == "raw"], "синтетика") +
                      eval_rules(ores[ores.prep.isin(["raw", "base"])], "Perestoroha"))
print(pp_tab.round(4).to_string(index=False))

# %% [markdown]
# ## 16. Структурування результату та JSON з метаданими
#
# З останнього рядка рукописної сторінки вилучаємо прізвище, групу і дату та перевіряємо їх
# правилами (дата існує і не в майбутньому, група за шаблоном). Для кожного розпізнавання
# зберігаємо інструмент, версію, параметри, впевненість, час, хеш зображення та EXIF.

# %%
FIELD_RE = {"date": r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b",
            "group": r"\b([А-ЯІЇЄҐ]{2,4})\s?[-–]\s?(\d{2,3})\b"}


def extract_fields(text):
    t = normalize(text)
    last = [l for l in str(text).splitlines() if l.strip()]
    last = last[-1] if last else ""
    out = {"surname": None, "group": None, "date": None}
    m = re.search(FIELD_RE["group"], t)
    out["group"] = f"{m.group(1)}-{m.group(2)}" if m else None
    m = re.search(FIELD_RE["date"], t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        try:
            dt = date(y, mo, d)
            out["date"] = dt.isoformat() if dt <= date.today() else None
        except ValueError:
            out["date"] = None
    m = re.match(r"\s*([А-ЯІЇЄҐ][а-яіїєґ'’]+)", last)
    out["surname"] = m.group(1) if m else None
    out["needs_review"] = any(v is None for v in out.values())
    return out


if len(own):
    records = []
    for r in own.itertuples():
        records.append({
            "file": r.file, "sha256": sha256_file(f"my_photos/{r.file}", 64),
            "exif": {"taken": r.taken, "camera": r.camera, "make": r.make},
            "engine": r.engine,
            "engine_version": TESS_VERSION if r.engine == "Tesseract" else f"{VLM_MODEL} ({(VLM_DIGEST or '')[:12]})",
            "params": ({"lang": TESS_LANG, "oem": 1, "psm": 6} if r.engine == "Tesseract"
                       else {"temperature": 0, "seed": SEED, "num_predict": 1024,
                             "prompt": "hist" if r.prep.endswith("hist") else "base"}),
            "preprocessing": r.prep, "confidence": round(float(r.conf), 4), "seconds": round(float(r.sec), 3),
            "cer": round(float(r.cer), 4), "wer": round(float(r.wer), 4),
            "text": r.pred,
            "fields": extract_fields(r.pred) if r.file.startswith("page") else None,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        })
    with open("results/own_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    gt_fields = {f: extract_fields(transcription(f))
                 for f in own.file.unique() if f.startswith("page")}
    print("Поля з еталону:", gt_fields)
    ex = [x for x in records if x["fields"]]
    for x in ex:
        ref = gt_fields[x["file"]]
        x["fields_correct"] = {k: x["fields"][k] == ref[k] for k in ["surname", "group", "date"]}
    print(pd.DataFrame([{"file": x["file"], "engine": x["engine"], "prep": x["preprocessing"], **x["fields"],
                         **{f"ok_{k}": v for k, v in x["fields_correct"].items()}} for x in ex]).to_string(index=False))
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:1500])

# %% [markdown]
# ## 17. Підсумкова таблиця результатів (CER, WER, час)

# %%
final = pd.concat([
    res.assign(group="синтетика (усі рівні)").groupby(["group", "config"]).agg(CER=("cer", "mean"), WER=("wer", "mean"), sec=("sec", "mean")),
    ores.assign(group="Perestoroha").groupby(["group", "config"]).agg(CER=("cer", "mean"), WER=("wer", "mean"), sec=("sec", "mean")),
] + ([own.assign(group="власні", config=own.engine + " / " + own.prep).groupby(["group", "config"])
      .agg(CER=("cer", "mean"), WER=("wer", "mean"), sec=("sec", "mean"))] if len(own) else []))
print(final.round(3).to_string())
final.round(4).to_csv("results/final_table.csv", lineterminator="\n")
