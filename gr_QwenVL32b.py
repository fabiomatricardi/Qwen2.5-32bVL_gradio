# from https://huggingface.co/spaces/ysharma/Microsoft_Phi-3-Vision-128k/blob/main/app.py
# https://qwenlm.github.io/blog/qwen2.5-vl-32b/

import gradio as gr
import base64
from openai import OpenAI
from PIL import Image
import io
from datetime import datetime
import random
import string

# Background of the Chatbot as a placeholder... really smart!
PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://www.ilinkandlink.com/wp-content/uploads/2025/02/20250205230750.webp" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Qwen 2.5 VL 32b Instruct</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Multimodal, Smarter and Lighter - Fine-grained Image Understanding and Reasoning</p>
</div>
"""

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    print(f'Logfile_{res}.md  CREATED')
    return f'Logfile_{res}.md'

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to encode the image using a Pillow Image object // not used here
def encode_image_pillow(image):
    # Create a bytes buffer
    buffer = io.BytesIO()
    # Save the image to the buffer
    image.save(buffer, format="PNG")  # You can use another format like "JPEG" if needed
    # Get the byte data from the buffer
    byte_data = buffer.getvalue()
    # Encode the byte data to base64
    base64_encoded = base64.b64encode(byte_data).decode("utf-8")
    return base64_encoded
# base64_string = encode_image_pillow(PILImage)

# CSS only to justify center the title
mycss = """
#warning {justify-content: center; text-align: center}
"""

logafilename = genRANstring(5)

with gr.Blocks(theme=gr.themes.Soft(secondary_hue=gr.themes.colors.orange,primary_hue=gr.themes.colors.blue),
               fill_width=True,css=mycss) as demo: #https://www.gradio.app/guides/theming-guide
        gr.Markdown("# Chat with Qwen 2.5 VL 32b Instruct - your Multi modal assistant",elem_id='warning')
        with gr.Row():
            with gr.Column(scale=1):
                genlogo = gr.Image('https://www.ilinkandlink.com/wp-content/uploads/2025/02/20250205230750.webp',
                                   show_label=False)
                gr.Markdown('### Remember the API Key')
                APIKey = gr.Textbox(value="", 
                            label="Open Router API key",
                            type='password',placeholder='Paste your API key',)
                gr.Markdown('### Tuning Parameters')
                maxlen = gr.Slider(minimum=250, maximum=4096, value=2048, step=1, label="Max new tokens")
                temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.45, step=0.1, label="Temperature")          
                log = gr.Markdown(logafilename, label='Log File name',container=True, show_label=True)
                botgr = gr.JSON(value=[],show_label=False,visible=False) #segregate chat message from presentation
                clear = gr.Button(value='Delete History',variant='secondary',
                                  icon='https://img.freepik.com/premium-vector/minimal-trash-bin-icon-vector_941526-16016.jpg')
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages",min_height='66vh',
                                     placeholder=PLACEHOLDER,
                                     show_copy_button = True,
                                     avatar_images=['https://i.ibb.co/PvqbDphL/user.png',
                                     'https://qwenlm.github.io/img/logo.png'],)
                msg = gr.MultimodalTextbox(interactive=True, file_types=["image"], 
                                           placeholder="Enter message or upload file...", 
                                           show_label=False)
                

                def clearData():
                    hiddenChat = gr.JSON(value=[],show_label=False,visible=False)
                    return hiddenChat

                def user(user_message, history, cbthst):
                    if  user_message['files']:
                        print('we have an image')
                        image = user_message['files'][-1]
                        text = user_message['text']
                        base64_image = encode_image(image)
                        logging = f'USER Image> {image}\nUSER text> {text}\n'
                        writehistory(logafilename,logging)
                        print(logging)
                        cbthst.append({"role": "user", "content": gr.Image(image, show_label=False, height=150)})
                        cbthst.append({"role": "user", "content": text})
                        history.append({"role": "user", "content": [
                                                {"type": "text","text": text},
                                                {"type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",}}]})
                    else:
                        print('no image')
                        text = user_message['text']
                        logging = f'USER text> {text}\n'
                        writehistory(logafilename,logging)
                        print(logging)
                        cbthst.append({"role": "user", "content": text})   #goes to the chatbox for presentation only
                        history.append({"role": "user", "content": text})  #goes to the API                           
                    return "", history, cbthst
                        

                def respond(chat_history, api,t,m,cbthst):
                    print(cbthst)
                    client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=api,)
                    stream = client.chat.completions.create(
                        extra_headers={},
                        extra_body={},
                        model="qwen/qwen2.5-vl-3b-instruct:free",     
                        messages=cbthst,
                        max_tokens=m,
                        stream=True,
                        temperature=t)
                    chat_history.append({"role": "assistant", "content": ""})
                    cbthst.append({"role": "assistant", "content": ""})
                    for chunk in stream:
                        chat_history[-1]['content'] += chunk.choices[0].delta.content
                        cbthst[-1]['content'] += chunk.choices[0].delta.content

                        yield chat_history, cbthst
                    logging = f"ASSISTANT> {chat_history[-1]['content']}\n"
                    writehistory(logafilename,logging)                   

        clear.click(clearData,[],[botgr])
        msg.submit(user, [msg, botgr,chatbot], [msg, botgr, chatbot]).then(
                respond, [chatbot,APIKey,temperature,maxlen,botgr], [chatbot,botgr])


# RUN THE MAIN
if __name__ == "__main__":
    demo.launch(inbrowser=True)

