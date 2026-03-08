import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import tempfile
import io


# --------------------------------------------------
# YOUR ORIGINAL COLAB FUNCTIONS (UNCHANGED)
# --------------------------------------------------

def preprocess_to_ink(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape)==3 else bgr
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2,2),np.uint8),
        iterations=1
    )

    return ink


def extract_components(ink, min_area=80, pad=2):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink,
        connectivity=8
    )

    comps=[]

    for i in range(1,num_labels):

        x,y,w,h,area = stats[i]

        if area < min_area:
            continue

        x0=max(0,x-pad)
        y0=max(0,y-pad)

        x1=min(ink.shape[1],x+w+pad)
        y1=min(ink.shape[0],y+h+pad)

        cx=x0+(x1-x0)/2
        cy=y0+(y1-y0)/2

        comps.append({
            "bbox":(x0,y0,x1-x0,y1-y0),
            "cx":cx,
            "cy":cy
        })

    return comps


def sort_reading_order(comps):

    heights=[c["bbox"][3] for c in comps]

    med_h=np.median(heights)

    y_thresh=max(18,int(med_h*0.8))

    lines=[]

    for c in sorted(comps,key=lambda x:x["cy"]):

        placed=False

        for line in lines:

            if abs(line["cy"]-c["cy"])<y_thresh:

                line["items"].append(c)
                line["cy"]=(line["cy"]*0.85 + c["cy"]*0.15)

                placed=True
                break

        if not placed:

            lines.append({"cy":c["cy"],"items":[c]})

    for line in lines:
        line["items"].sort(key=lambda x:x["cx"])

    lines.sort(key=lambda l:l["cy"])

    ordered=[]

    for line in lines:
        ordered.extend(line["items"])

    return ordered


def merge_dots(comps):

    ws=[c["bbox"][2] for c in comps]
    hs=[c["bbox"][3] for c in comps]

    med_w,med_h=np.median(ws),np.median(hs)

    dot_area_max=int((med_w*med_h)*0.18)

    dot_wh_max=int(max(12,min(med_w,med_h)*0.55))

    x_match_max=int(max(18,med_w*0.8))

    dots=[]
    stems=[]

    for c in comps:

        x,y,w,h=c["bbox"]

        if w*h<=dot_area_max and w<=dot_wh_max and h<=dot_wh_max:
            dots.append(c)
        else:
            stems.append(c)

    used=set()
    merged=[]

    for s in stems:

        sx,sy,sw,sh=s["bbox"]
        sxc=s["cx"]

        best=None
        best_dx=1e9

        for i,d in enumerate(dots):

            if i in used:
                continue

            dx,dy,dw,dh=d["bbox"]

            if dy+dh>sy:
                continue

            dxc=d["cx"]

            dist=abs(dxc-sxc)

            if dist<best_dx:
                best_dx=dist
                best=i

        if best is not None and best_dx<=x_match_max:

            used.add(best)

            d=dots[best]

            x1=min(sx,d["bbox"][0])
            y1=min(sy,d["bbox"][1])

            x2=max(sx+sw,d["bbox"][0]+d["bbox"][2])
            y2=max(sy+sh,d["bbox"][1]+d["bbox"][3])

            s["mbox"]=(x1,y1,x2-x1,y2-y1)

        else:
            s["mbox"]=s["bbox"]

        merged.append(s)

    return merged


def normalize_glyph_from_bbox(ink_full,bbox,target_h=110):

    x,y,w,h=bbox

    crop=ink_full[y:y+h,x:x+w]

    scale=target_h/max(h,1)

    nw=max(1,int(w*scale))

    resized=cv2.resize(crop,(nw,target_h))

    rgba=np.zeros((target_h,nw,4),dtype=np.uint8)

    rgba[:,:,3]=resized

    return Image.fromarray(rgba,"RGBA")


# --------------------------------------------------
# GLYPH BANK
# --------------------------------------------------

def build_glyph_bank(img_path,sequence,min_area):

    bgr=cv2.imread(img_path)

    ink=preprocess_to_ink(bgr)

    comps=extract_components(ink,min_area)

    comps=merge_dots(comps)

    comps=sort_reading_order(comps)

    bank={}
    debug=[]

    n=min(len(comps),len(sequence))

    for i in range(n):

        bbox=comps[i].get("mbox",comps[i]["bbox"])

        glyph=normalize_glyph_from_bbox(ink,bbox)

        if glyph:
            bank.setdefault(sequence[i],[]).append(glyph)

        debug.append({"char":sequence[i],"box":bbox})

    return bank,debug,bgr


# --------------------------------------------------
# RENDER TEXT
# --------------------------------------------------

def render_text_page(text,glyph_bank,letter_space,word_space,line_gap,target_h):

    page=Image.new("RGBA",(1600,2000),(255,255,255,255))

    x=120
    y=120

    for line in text.split("\n"):

        for word in line.split(" "):

            for ch in word:

                if ch not in glyph_bank:
                    continue

                g=random.choice(glyph_bank[ch])

                scale=target_h/max(g.size[1],1)

                nw=int(g.size[0]*scale)

                g=g.resize((nw,target_h))

                page.alpha_composite(g,(x,y))

                x+=nw+letter_space

            x+=word_space

        x=120
        y+=target_h+line_gap

    return page.convert("RGB")


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

def handwriting_interface():

    st.title("Smart Handwriting Generator")

    st.warning("""
    **Alphabet Image Instructions**

    Write characters exactly in this order:

    CAPITAL LETTERS first:
    A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

    Then small letters:
    a b c d e f g h i j k l m n o p q r s t u v w x y z
    """)

    st.warning("""
    **Digit & Punctuation Image Instructions**

    Write characters in TWO LINES exactly like this:

    0 1 2 3 4 5 6 7 8 9
    . , ; : ? ! ( ) - +
    """)

    col1,col2=st.columns([1,1])

    with col1:

        alpha_file=st.file_uploader("Upload Alphabet Image *")

        digit_file=st.file_uploader("Upload Digit & Punctuation Image *")

        word_file=st.file_uploader("Upload Word Sample (letter spacing) *")

        sent_file=st.file_uploader("Upload Sentence Sample (word spacing) *")

        para_file=st.file_uploader("Upload Paragraph Sample (line spacing) *")

    glyph_bank={}

    with col2:

        if alpha_file:

            tmp=tempfile.NamedTemporaryFile(delete=False)
            tmp.write(alpha_file.read())

            alpha_seq=list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

            bank,debug,img=build_glyph_bank(tmp.name,alpha_seq,120)

            glyph_bank.update(bank)

            debug_img=img.copy()

            for d in debug:

                x,y,w,h=d["box"]

                cv2.rectangle(debug_img,(x,y),(x+w,y+h),(255,0,0),2)

                cv2.putText(debug_img,d["char"],(x,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            st.image(debug_img,caption="Alphabet Debug View")

        if digit_file:

            tmp=tempfile.NamedTemporaryFile(delete=False)

            tmp.write(digit_file.read())

            digit_seq=list("0123456789.,;:?!()-+")

            bank,debug,img=build_glyph_bank(tmp.name,digit_seq,60)

            glyph_bank.update(bank)

            debug_img=img.copy()

            for d in debug:

                x,y,w,h=d["box"]

                cv2.rectangle(debug_img,(x,y),(x+w,y+h),(255,0,0),2)

                cv2.putText(debug_img,d["char"],(x,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            st.image(debug_img,caption="Digit Debug View")

    st.divider()

    speech_text=""
    if "sentence_queue" in st.session_state:
        speech_text=" ".join(st.session_state.sentence_queue)

    text_input = st.text_area(
        "Edit Text Before Generating",
        value=speech_text,
        height=150
    )

    if st.button("Generate"):

        if not all([alpha_file,digit_file,word_file,sent_file,para_file]):
            st.error("Please upload ALL 5 images before generating handwriting.")
            return

        img=render_text_page(text_input,glyph_bank,8,30,50,100)

        st.image(img)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        st.download_button(
            label="Download Handwriting Image",
            data=buf.getvalue(),
            file_name="handwriting_output.png",
            mime="image/png"
        )