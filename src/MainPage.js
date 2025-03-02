import React ,  { useState, useEffect } from 'react';
import VideoControl from './VideoControl';
import AudioChatResponse from './AudioChatResponse';
import './MainPage.css';

const MainPage = () => {
    const [replyText, setReplyText] = useState('');


     // Fetch or mock reply when the component loads
     useEffect(() => {
        const mockResponse = {
            generated_reply: "Hi there! I'm here to help. Don't worry, everything is okay. Would you like me to tell you where you are?"
        };

        setReplyText(mockResponse.generated_reply);
    }, []);  // Empty dependency array means this runs only once on page load


    return (
        <div className="main-page">
            <nav className="main-menu">
                <h2>SamurAI</h2>
                <div className="menu-links">
                    <a href="#">Home</a>
                    <a href="#">About</a>
                    <a href="#">Contact</a>
                </div>
            </nav>
            <VideoControl />

            <AudioChatResponse replyText={replyText} />
            
        </div>
    );
};

const styles = {

    button: { padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px', margin: '20px 0' },
};


export default MainPage;