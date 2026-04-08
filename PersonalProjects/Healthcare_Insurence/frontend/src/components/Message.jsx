// src/components/Message.jsx
const Message = ({ text, sender }) => {
    const isUser = sender === 'user';

    return (
        <div style={{
            display: 'flex',
            justifyContent: isUser ? 'flex-end' : 'flex-start',
            margin: '10px 0'
        }}>
            <div style={{
                backgroundColor: isUser ? '#007bff' : '#e9e9eb',
                color: isUser ? 'white' : 'black',
                padding: '10px 15px',
                borderRadius: '15px',
                maxWidth: '70%'
            }}>
                {text}
            </div>
        </div>
    );
};

export default Message;