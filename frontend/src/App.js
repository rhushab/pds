import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import {
  Container,
  Box,
  Typography,
  Button,
  CircularProgress,
  TextField,
  Paper,
} from '@mui/material';
import { makeStyles } from '@mui/styles';

const useStyles = makeStyles((theme) => ({
  container: {
    textAlign: 'center',
    padding: '20px',
  },
  title: {
    fontSize: '2.5em',
    color: '#000',
    marginBottom: '20px',
  },
  hiddenInput: {
    display: 'none',
  },
  fileInputLabel: {
    backgroundColor: '#000',
    color: '#fff',
    padding: '10px 20px',
    fontSize: '1em',
    cursor: 'pointer',
    borderRadius: '5px',
    margin: '10px 5px',
    display: 'inline-block',
    '&:hover': {
      backgroundColor: '#444',
    },
  },
  button: {
    backgroundColor: '#000 !important',
    color: '#fff !important',
    padding: '10px 20px',
    borderRadius: '5px',
    fontSize: '1em',
    cursor: 'pointer',
    margin: '10px 5px',
    '&:hover': {
      backgroundColor: '#444 !important',
    },
    '&:disabled': {
      backgroundColor: '#999 !important',
      cursor: 'not-allowed',
    },
  },
  textInput: {
    margin: '10px 0',
    width: '300px',
  },
  chatContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  chatBox: {
    width: '100%',
    maxWidth: '600px',
    height: '400px',
    padding: '10px',
    overflowY: 'scroll',
    backgroundColor: '#fff',
    marginBottom: '10px',
  },
  chatMessage: {
    padding: '10px',
    margin: '5px 0',
    borderRadius: '5px',
  },
  userMessage: {
    backgroundColor: '#e1e1e1',
    textAlign: 'right',
  },
  botMessage: {
    backgroundColor: '#d1d1d1',
    textAlign: 'left',
  },
  spinner: {
    margin: '20px auto',
  },
  statusMessage: {
    fontSize: '1.2em',
  },
  fileNameDisplay: {
    margin: '10px 0',
    fontStyle: 'italic',
  },
}));

function App() {
  const classes = useStyles();
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [statusMessage, setStatusMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState('');
  const [loading, setLoading] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);

  const handleFileChange = (event) => {
    setFiles(event.target.files);
    if (event.target.files.length > 0) {
      setSelectedFile(event.target.files[0].name);
    } else {
      setSelectedFile('');
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setStatusMessage('Please select a file before uploading.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    try {
      const response = await axios.post(
        'http://localhost:5000/upload',
        formData
      );
      setStatusMessage(response.data.message);
      setMessages([]);
      setIsUploaded(true);
    } catch (error) {
      setStatusMessage('Error uploading files. Please try again.');
    }
    setLoading(false);
  };

  const handleQuery = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/query', {
        question,
        filename: selectedFile,
      });
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'user', text: question },
        { sender: 'bot', text: response.data.answer },
      ]);
      setQuestion('');
    } catch (error) {
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'user', text: question },
        { sender: 'bot', text: 'Error retrieving answer. Please try again.' },
      ]);
      setQuestion('');
    }
    setLoading(false);
  };

  const handleNewUpload = () => {
    setFiles([]);
    setQuestion('');
    setMessages([]);
    setStatusMessage('');
    setSelectedFile('');
    setIsUploaded(false);
  };

  return (
    <Container maxWidth="md" className={classes.container}>
      <Typography variant="h1" component="h1" className={classes.title}>
        {isUploaded ? 'Query' : 'Upload PDF'}
      </Typography>
      {isUploaded ? (
        <Box className={classes.chatContainer}>
          <Paper elevation={3} className={classes.chatBox}>
            {messages.map((message, index) => (
              <Box
                key={index}
                className={`${classes.chatMessage} ${
                  message.sender === 'user'
                    ? classes.userMessage
                    : classes.botMessage
                }`}
              >
                <Typography>{message.text}</Typography>
              </Box>
            ))}
          </Paper>
          {loading && <CircularProgress className={classes.spinner} />}
          <TextField
            variant="outlined"
            fullWidth
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question"
            className={classes.textInput}
            margin="normal"
          />
          <Button
            variant="contained"
            onClick={handleQuery}
            disabled={loading}
            className={classes.button}
          >
            Submit
          </Button>
          <Button
            variant="contained"
            onClick={handleNewUpload}
            className={classes.button}
          >
            New Upload
          </Button>
        </Box>
      ) : (
        <Box>
          <input
            type="file"
            multiple
            onChange={handleFileChange}
            id="file-input"
            className={classes.hiddenInput}
          />
          <label htmlFor="file-input" className={classes.fileInputLabel}>
            Choose Files
          </label>
          {selectedFile && (
            <Typography className={classes.fileNameDisplay}>
              Selected file: {selectedFile}
            </Typography>
          )}
          {loading && <CircularProgress className={classes.spinner} />}
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={loading || files.length === 0}
            className={classes.button}
          >
            Upload
          </Button>
          <Typography className={classes.statusMessage}>
            {statusMessage}
          </Typography>
        </Box>
      )}
    </Container>
  );
}

export default App;
