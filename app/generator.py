import logging
import json
import asyncio
from pydantic import BaseModel, Field
from app.config import config
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    answer: str = Field(description="The final answer written in the language of the query. Must be grounded strictly in the context.")
    grounded: bool = Field(description="True if the answer is fully grounded in the retrieved passages, False otherwise.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 indicating how certain we are of the groundedness.")
    refusal: bool = Field(description="True if the query cannot be answered based on the context, is off-topic, or contains unsafe themes.")
    refusal_reason: str = Field(description="Brief explanation in English why the query was refused (empty if refusal is False).")
    citations: list[str] = Field(description="List of URLs or source labels from the retrieved passages that support this answer.")

# Simulated responses for offline/keyless testing
SIMULATED_ANSWERS = {
    "hi": "रक्तचाप मापने के लिए सटीक रीडिंग प्राप्त करने की निम्नलिखित प्रक्रिया है:\n1. रक्तचाप मापने से 30 मिनट पहले कैफीन का सेवन न करें और व्यायाम करने से बचें।\n2. जांच शुरू होने से कम से कम 5 मिनट पहले शांत बैठें।\n3. जांच के दौरान कुर्सी पर सीधे बैठें, पैर जमीन पर सपाट रखें और अपनी बांह को हृदय के स्तर पर सहारा दें।\n4. कफ आपकी ऊपरी बांह के कम से कम 80% हिस्से को ढकना चाहिए।\n5. 1-2 मिनट के अंतराल पर कम से कम दो माप लें और उनका औसत निकालें।",
    "bn": "রক্তচাপ পরিমাপ করার সময় সঠিক রিডিং নিশ্চিত করার উপায়গুলি হলো:\n১. রক্তচাপ পরিমাপের ৩০ মিনিট আগে ক্যাফেইনযুক্ত পানীয় পান বা ব্যায়াম করবেন না।\n২. পরীক্ষা শুরু করার আগে ৫ মিনিট শান্ত হয়ে বসুন।\n৩. পরিমাপের সময় মেরুদণ্ড সোজা করে চেয়ারে বসুন, পা মেঝেতে সমান্তরাল রাখুন এবং বাহুটি হৃৎপিণ্ডের স্তরের কাছাকাছি রাখুন।\n৪. কাফের স্ফীত অংশ আপনার বাহুর অন্তত ৮০% অংশ ঢেকে রাখা উচিত।\n৫. অন্তত দুটি পরিমাপ নিয়ে ফলাফল গড় করুন।",
    "en": "To ensure accurate blood pressure measurement:\n1. Avoid caffeine, exercise, and smoking 30 minutes before measuring.\n2. Sit quietly for 5 minutes before taking a reading.\n3. Sit upright in a chair with feet flat on the floor and support your arm at heart level.\n4. Ensure cuff covers 80% of your upper arm.\n5. Take two readings 1-2 minutes apart and average them.",
    "ta": "துல்லியமான இரத்த அழுத்த அளவீட்டை உறுதி செய்ய:\n1. 30 நிமிடங்களுக்கு முன் காஃபின் உட்கொள்ள வேண்டாம்.\n2. 5 நிமிடங்கள் அமைதியாக உட்காரவும்.\n3. கையை இதய மட்டத்தில் வைத்து நேராக உட்காரவும்.",
    "te": "ఖచ్చితమైన రక్తపోటు కొలతను నిర్ధారించడానికి:\n1. 30 నిమిషాల ముందు కెఫిన్ తీసుకోకండి.\n2. 5 నిమిషాలు ప్రశాంతంగా కూర్చోండి.\n3. చేతిని గుండె స్థాయిలో నిటారుగా కూర్చోండి.",
    "mr": "अचूक रक्तदाब मोजणीसाठी:\n१. मोजण्यापूर्वी ३० मिनिटे कॅफिन टाळा.\n२. ५ मिनिटे शांत बसा.\n३. हात हृदयाच्या पातळीवर ठेवून सरळ बसा.",
    "gu": "ચોક્કસ બ્લડ પ્રેશર રીડિંગ મેળવવા માટે:\n૧. ૩૦ મિનિટ પહેલા કેફીન ટાળો.\n૨. ૫ મિનિટ શાંતિથી બેસો.\n૩. હાથને હૃદયના સ્તરે રાખીને સીધા બેસો.",
    "kn": "ನಿಖರವಾದ ರಕ್ತದೊತ್ತಡ ಮಾಪನಕ್ಕಾಗಿ:\n1. 30 ನಿಮಿಷಗಳ ಮೊದಲು ಕೆಫೀನ್ ಸೇವಿಸಬೇಡಿ.\n2. 5 ನಿಮಿಷಗಳ ಕಾಲ ಶಾಂತವಾಗಿ ಕುಳಿತುಕೊಳ್ಳಿ.",
    "ml": "കൃത്യമായ രക്തസമ്മർദ്ദം അളക്കുന്നതിന്:\n1. 30 മിനിറ്റ് മുമ്പ് കഫീൻ ഒഴിവാക്കുക.\n2. 5 മിനിറ്റ് ശാന്തമായി ഇരിക്കുക.",
    "pa": "ਸਹੀ ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ ਮਾਪਣ ਲਈ:\n1. 30 ਮਿੰਟ ਪਹਿਲਾਂ ਕੈਫੀਨ ਨਾ ਲਓ।\n2. 5 ਮਿੰਟ ਸ਼ਾਂਤ ਬੈਠੋ।",
    "od": "ସଠିକ୍ ରକ୍ତଚାପ ମାପ ପାଇଁ:\n୧. ୩୦ ମିନିଟ୍ ପୂର୍ବରୁ କ୍ୟାଫିନ୍ ଏଡାନ୍ତୁ।\n୨. ୫ ମିନିଟ୍ ଶାନ୍ତ ହୋଇ ବସନ୍ତୁ。"
}


REFUSAL_MESSAGES = {
    "hi": "मुझे खेद है, लेकिन प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
    "bn": "আমি দুঃখিত, কিন্তু প্রদত্ত প্রসঙ্গে এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।",
    "en": "I am sorry, but there is not enough information in the provided context to answer this question.",
    "ta": "மன்னிக்கவும், வழங்கப்பட்ட சூழலில் இந்த கேள்விக்கு பதிலளிக்க போதுமான தகவல்கள் இல்லை.",
    "te": "క్షమించండి, అందించిన సందర్భంలో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి సరిపోయే సమాచారం లేదు.",
    "mr": "मला माफ करा, परंतु दिलेल्या संदर्भात या प्रश्नाचे उत्तर देण्यासाठी पुरेशी माहिती नाही.",
    "gu": "મને માફ કરશો, પરંતુ આપેલા સંદર્ભમાં આ પ્રશ્નનો જવાબ આપવા માટે પૂરતી માહિતી નથી.",
    "kn": "ಕ್ಷಮಿಸಿ, ನೀಡಲಾದ ಸಂದರ್ಭದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿಯಿಲ್ಲ.",
    "ml": "ക്ഷമിക്കണം, നൽകിയിട്ടുള്ള സന്ദർഭത്തിൽ ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങളില്ല.",
    "pa": "ਮੈਨੂੰ ਅਫਸੋਸ ਹੈ, ਪਰ ਦਿੱਤੇ ਗਏ ਸੰਦਰਭ ਵਿੱਚ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਲਈ ਕਾਫੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
    "od": "ଦୁଃଖିତ, ପ୍ରଦତ୍ତ ପ୍ରସଙ୍ଗରେ ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବା ପାଇଁ ପର୍ଯ୍ୟାପ୍ତ ସୂଚନା ନାହିଁ ।"
}

async def generate_answer(query: str, retrieved_passages: list[dict], language: str = "hi", provider: str = "gemini") -> RAGResponse:
    """
    Generates a grounded response based on the query and retrieved passages.
    Employs structured JSON schema output and retries.
    If no key is present, falls back to simulated offline responses.
    """
    
    # 1. Fallback to offline simulated generation if no keys are configured
    if not config.GEMINI_API_KEY and not config.GROQ_API_KEY:
        logger.warning("No LLM API keys set. Running in offline simulated LLM mode.")
        await asyncio.sleep(0.05)  # Simulate small network tick
        
        q_lower = query.lower()
        citations = [p.get("url", "https://www.cdc.gov/health") for p in retrieved_passages if p.get("url")]
        
        if "blood pressure" in q_lower or "रक्तचाप" in q_lower or "রক্তচাপ" in q_lower:
            ans_text = SIMULATED_ANSWERS.get(language, SIMULATED_ANSWERS["en"])
        elif "water" in q_lower or "पानी" in q_lower or "जल" in q_lower:
            ans_text = {
                "en": "Healthy adults should drink around 8 to 10 glasses (about 2 to 2.5 liters) of water daily to stay hydrated.",
                "hi": "एक स्वस्थ वयस्क को प्रतिदिन लगभग 8 से 10 गिलास (लगभग 2 से 2.5 लीटर) पानी पीना चाहिए।",
                "bn": "একজন সুস্থ প্রাপ্তবয়স্কের প্রতিদিন প্রায় ৮ থেকে ১০ গ্লাস (প্রায় ২ থেকে ২.৫ লিটার) জল পান করা উচিত।"
            }.get(language, "Healthy adults should drink around 8 to 10 glasses of water daily to stay hydrated.")
        elif "temperature" in q_lower or "तापमान" in q_lower or "तापমাত্রা" in q_lower or "fever" in q_lower:
            ans_text = {
                "en": "The normal average human body temperature is typically 98.6°F (37°C).",
                "hi": "मानव शरीर का सामान्य औसत तापमान 98.6°F (37°C) होता है।",
                "bn": "মানুষের শরীরের স্বাভাবিক গড় তাপমাত্রা ৯৮.৬° ফারেনহাইট (৩৭° সেলসিয়াস)।"
            }.get(language, "The normal human body temperature is 98.6°F (37°C).")
        elif retrieved_passages and len(retrieved_passages) > 0:
            passage_snippet = " ".join([p.get("text", "") for p in retrieved_passages[:2]])
            if language == "en":
                ans_text = f"According to retrieved MSMARCO context: {passage_snippet}"
            elif language == "bn":
                ans_text = f"প্রদত্ত তথ্য অনুযায়ী: {passage_snippet}"
            else:
                ans_text = f"प्रदान किए गए संदर्भ के अनुसार: {passage_snippet}"
        else:
            refusal_text = REFUSAL_MESSAGES.get(language, REFUSAL_MESSAGES["hi"])
            return RAGResponse(
                answer=refusal_text,
                grounded=False,
                confidence=0.0,
                refusal=True,
                refusal_reason="Insufficient context for query in offline mode",
                citations=[]
            )
            
        return RAGResponse(
            answer=ans_text,
            grounded=True,
            confidence=0.92,
            refusal=False,
            refusal_reason="",
            citations=citations[:2] if citations else ["https://www.cdc.gov/health"]
        )


    # 2. Prepare prompt context
    context_str = ""
    for idx, passage in enumerate(retrieved_passages):
        url_label = passage.get("url", f"Document {idx + 1}")
        context_str += f"Source: {url_label}\nContent: {passage['text']}\n\n"
        
    prompt = f"""
You are a highly precise grounded Q&A model. Your task is to answer the user query based ONLY on the provided context.
Do not make assumptions, expand beyond the context, or introduce outside knowledge.

Context:
{context_str}

User Query: {query}
Target Language: {language}

Instructions:
1. Analyze the context and query carefully.
2. If the context does not contain enough information to answer the query, set 'refusal' to True, and provide a polite refusal text in the Target Language (e.g. Hindi or Bengali) in the 'answer' field.
3. If the query is off-topic (unrelated to general knowledge/facts or the context), set 'refusal' to True.
4. If you find the answer in the context, write a complete, clear answer in the Target Language. Set 'grounded' to True and 'refusal' to False.
5. In the 'citations' field, return the Source URLs/labels that directly support your answer.
"""

    # 3. Choose provider and execute
    if provider == "gemini" and config.GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            
            # Using structured schema generation
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RAGResponse,
                    temperature=0.1
                ),
            )
            
            res_text = response.text
            res_dict = json.loads(res_text)
            return RAGResponse(**res_dict)
            
        except Exception as e:
            logger.exception("Error calling Gemini API")
            if config.GROQ_API_KEY:
                logger.info("Attempting fallback to Groq...")
                return await _generate_answer_groq(prompt, query, language)
            raise e
            
    elif provider == "groq" and config.GROQ_API_KEY:
        return await _generate_answer_groq(prompt, query, language)
        
    else:
        # Configuration mismatch or fallback trigger
        raise ValueError("Invalid LLM configuration or missing API keys")

async def _generate_answer_groq(prompt: str, query: str, language: str) -> RAGResponse:
    """Fallback generator using Groq's JSON mode."""
    from groq import Groq
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        
        # Groq system message instructing structured output
        system_msg = (
            "You are a helpful assistant that outputs JSON matching this schema: "
            "{\n"
            "  \"answer\": \"string (response in query language)\",\n"
            "  \"grounded\": boolean,\n"
            "  \"confidence\": float,\n"
            "  \"refusal\": boolean,\n"
            "  \"refusal_reason\": \"string\",\n"
            "  \"citations\": [\"string\"]\n"
            "}"
        )
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=10.0
        )
        
        res_text = completion.choices[0].message.content
        res_dict = json.loads(res_text)
        return RAGResponse(**res_dict)
        
    except Exception as e:
        logger.exception("Error calling Groq API")
        raise e
