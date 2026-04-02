// src/components/ChatWindow.jsx
import { useState, useEffect, useRef } from 'react';
import Message from './Message';

const ChatWindow = () => {
    const [messages, setMessages] = useState([
        { text: "Hello! I am your Intelligent Policy Assistant. How can I help you today?", sender: 'bot' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Auto-scroll to bottom when new messages arrive
    const scrollRef = useRef(null);
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isLoading]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userText = input;
        setInput(''); // Clear input immediately for better UX
        setMessages((prev) => [...prev, { text: userText, sender: 'user' }]);
        setIsLoading(true);

        try {
            const response = await fetch("http://localhost:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: userText }),
            });

            if (!response.ok) {
                throw new Error("Backend is not responding properly.");
            }

            const data = await response.json();

            // Format the bot's response with citations
            let botResponse = data.answer;
            if (data.sources && data.sources.length > 0) {
                botResponse += `\n\n📚 Sources: ${data.sources.join(', ')}`;
            }

            setMessages((prev) => [...prev, { text: botResponse, sender: 'bot' }]);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                { text: "Sorry, I'm having trouble connecting to the policy engine. Is the backend running?", sender: 'bot' }
            ]);
            console.error("Error:", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{
            maxWidth: '700px',
            margin: '40px auto',
            border: '1px solid #e0e0e0',
            borderRadius: '12px',
            display: 'flex',
            flexDirection: 'column',
            height: '80vh',
            backgroundColor: '#fff',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
            {/* Chat History */}
            <div
                ref={scrollRef}
                style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column'
                }}
            >
                {messages.map((msg, index) => (
                    <Message key={index} text={msg.text} sender={msg.sender} />
                ))}
                {isLoading && (
                    <div style={{ fontStyle: 'italic', color: '#888', margin: '10px 0' }}>
                        Assistant is searching policies...
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div style={{
                padding: '20px',
                borderTop: '1px solid #eee',
                display: 'flex',
                gap: '10px'
            }}>
                <input
                    type="text"
                    value={input}
                    disabled={isLoading}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Ask about accidental disability..."
                    style={{
                        flex: 1,
                        padding: '12px 15px',
                        borderRadius: '8px',
                        border: '1px solid #ccc',
                        fontSize: '16px',
                        outline: 'none'
                    }}
                />
                <button
                    onClick={handleSend}
                    disabled={isLoading}
                    style={{
                        padding: '0 20px',
                        backgroundColor: isLoading ? '#ccc' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold'
                    }}
                >
                    Send
                </button>
            </div>
        </div>
    );
};

export default ChatWindow;