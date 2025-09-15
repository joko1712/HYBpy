import React, { useState } from "react";
import { auth } from "../firebase-config";
import {
    createUserWithEmailAndPassword,
    sendEmailVerification,
} from "firebase/auth";
import Avatar from "@mui/material/Avatar";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Grid";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import {
    AppBar as MuiAppBar,
    Box,
    Toolbar,
    Typography,
    IconButton,
    Container,
    Link,
    Button,
} from "@mui/material";
import logo from "../Image/HYBpyINVIS_logo.png";
import { handleContactUsClick } from "./ContactUs";
import EmailIcon from '@mui/icons-material/Email';



const defaultTheme = createTheme();

const AppBar = styled(MuiAppBar)(({ theme }) => ({
    zIndex: theme.zIndex.drawer + 1,
}));

function Register() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [error, setError] = useState("");
    const [emailError, setEmailError] = useState("");
    const navigate = useNavigate();

    const validateEmail = (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    };

    const handleEmailChange = (e) => {
        const emailValue = e.target.value;
        setEmail(emailValue);
        if (!validateEmail(emailValue) && emailValue) {
            setEmailError("Invalid email format.");
        } else {
            setEmailError("");
        }
    };

    const registerUser = (e) => {
        e.preventDefault();
        setError("");

        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            window.alert("Passwords do not match.");
            return;
        }

        if (password.length < 10) {
            setError("Password must be at least 10 characters long.");
            window.alert("Password must be at least 10 characters long.");
            return;
        }

        if (!validateEmail(email)) {
            window.alert(
                "Invalid email format. Please enter a valid email address."
            );
            return;
        }

        createUserWithEmailAndPassword(auth, email, password)
            .then((res) => {
                sendEmailVerification(auth.currentUser)
                    .then(() => {
                        window.alert(
                            "A confirmation link has been sent to your email. Please verify your email address."
                        );
                        navigate("/");
                    })
                    .catch((verificationError) => {
                        console.error("Verification error:", verificationError);
                        setError(verificationError.message);
                    });
                console.log(res.user);
            })
            .catch((err) => {
                setError(err.message);
            });

        setEmail("");
        setPassword("");
        setConfirmPassword("");
    };

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    minHeight: "100%",
                }}>
                <CssBaseline />
                <AppBar position='fixed' color='primary'>
                    <Toolbar sx={{ pr: "2px" }}>
                        <IconButton edge='start' color='inherit' size='small'>
                            <img
                                src={logo}
                                alt='logo'
                                width='200'
                                height='75'
                            />
                        </IconButton>
                    </Toolbar>
                </AppBar>

                <Container
                    component='main'
                    sx={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        mt: 10,
                        flexGrow: 1,
                        width: "100%",
                    }}>
                    <Box
                        sx={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            mt: 2,
                            maxWidth: "50%"
                        }}>
                        <h1>What is HYBpy?</h1>
                        <p>
                            HYBpy is designed to simplify the construction and
                            analyses hybrid models of{" "}
                            <Link href='https://www.sciencedirect.com/science/article/pii/S0098135422002897?via%3Dihub#abs0001'>
                                bioprocesses
                            </Link>{" "}
                            and{" "}
                            <Link href='https://www.mdpi.com/2673-2688/4/1/14#B25-ai-04-00014'>
                                biological systems
                            </Link>
                            . You can also install HYBpy on Windows to run
                            locally. Please visit the{" "}
                            <Link href='https://github.com/joko1712/HYBpy'>
                                HYBpy GitHub
                            </Link>{" "}
                            page.
                        </p>
                        <Box sx={{ mt: "5%" }} />
                        <Avatar sx={{ m: 1, bgcolor: "secondary.main" }}>
                            <LockOutlinedIcon />
                        </Avatar>
                        <Typography component='h1' variant='h5'>
                            Sign up
                        </Typography>
                        <Box
                            component='form'
                            noValidate
                            onSubmit={registerUser}
                            sx={{ mt: 1, maxWidth: "75%" }}>
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <TextField
                                        error={!!emailError}
                                        required
                                        fullWidth
                                        id='email'
                                        label='Email Address'
                                        name='email'
                                        autoComplete='email'
                                        value={email}
                                        onChange={handleEmailChange}
                                    />
                                </Grid>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        fullWidth
                                        name='password'
                                        label='Password'
                                        type='password'
                                        id='password'
                                        autoComplete='new-password'
                                        value={password}
                                        onChange={(e) =>
                                            setPassword(e.target.value)
                                        }
                                    />
                                </Grid>
                                <Grid item xs={12}>
                                    <TextField
                                        required
                                        fullWidth
                                        name='confirmPassword'
                                        label='Confirm Password'
                                        type='password'
                                        id='confirmPassword'
                                        autoComplete='new-password'
                                        value={confirmPassword}
                                        onChange={(e) =>
                                            setConfirmPassword(e.target.value)
                                        }
                                    />
                                </Grid>
                            </Grid>
                            {error && (
                                <Typography color='error' sx={{ mt: 2 }}>
                                    {error}
                                </Typography>
                            )}
                            <Button
                                type='submit'
                                fullWidth
                                variant='contained'
                                sx={{ mt: 3, mb: 2 }}>
                                Sign Up
                            </Button>
                            <Grid container justifyContent='flex-end'>
                                <Grid item>
                                    <Link href='/' variant='body2'>
                                        Already have an account? Sign in
                                    </Link>
                                </Grid>
                            </Grid>
                        </Box>
                    </Box>
                </Container>

                <Box
                    component='footer'
                    sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "1em",
                        backgroundColor: "#f1f1f1",
                        width: "100%",
                        position: "fixed",
                        bottom: 0,
                        left: 0,
                    }}>
                    <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
                    
                        <Typography
                            variant='body2'
                            align='center'
                            sx={{ flexGrow: 1 }}>
                            &copy; {new Date().getFullYear()} NOVA School of Science and Technology, Universidade NOVA de Lisboa. All rights
                            reserved.
                        </Typography>

                        <Button
                            color="inherit"
                            variant="text"
                            onClick={handleContactUsClick}
                            style={{ marginLeft: "1em", textTransform: "none", paddingRight: "2em" }}
                            startIcon={<EmailIcon />}
                        >
                            Contact Us
                        </Button>
                    </div>

                    <a
                        href="https://www.fct.unl.pt/en"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <img
                        src="https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png"
                        width="75px"
                        alt="FCT Logo"
                        style={{ marginLeft: "1em" }}
                        />
                    </a>
                </Box>
            </Box>
        </ThemeProvider>
    );
}

export default Register;
