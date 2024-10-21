import React, { useState } from "react";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Link from "@mui/material/Link";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { auth, signInWithGoogle } from "../firebase-config";
import { signInWithEmailAndPassword } from "firebase/auth";
import GoogleIcon from "@mui/icons-material/Google";
import { Route } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { Alert } from "@mui/material";

const defaultTheme = createTheme();

function Login({ manualSetCurrentUser }) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const loginUser = async (e) => {
        e.preventDefault();
        setError("");

        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            if (userCredential.user.emailVerified) {
                console.log("Email is verified");
                manualSetCurrentUser(userCredential.user);
                navigate("/");
            } else {
                console.log("Email is not verified");
                window.alert("Please verify your email address.");
            }
        } catch (error) {
            console.error("Login error:", error);
            setError(error.message);
            window.alert("Invalid email or password.");
        }

        setEmail("");
        setPassword("");
    };

    const loginWithGoogle = () => {
        setError("");

        signInWithGoogle(auth)
            .then((res) => {
                console.log(res.user);
            })
            .catch((err) => setError(err.message));
    };

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box
                sx={{
                    minHeight: "100vh",
                    display: "flex",
                    flexDirection: "column",
                }}>
                <Container component='main' maxWidth='xs' sx={{ flex: 1 }}>
                    <CssBaseline />
                    <Box
                        sx={{
                            marginTop: 8,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                        }}>
                        <Avatar sx={{ m: 1, bgcolor: "secondary.main" }}>
                            <LockOutlinedIcon />
                        </Avatar>
                        <Typography component='h1' variant='h5'>
                            Sign in
                        </Typography>
                        <Box component='form' onSubmit={loginUser} noValidate sx={{ mt: 1 }}>
                            <TextField
                                margin='normal'
                                required
                                fullWidth
                                id='email'
                                label='Email Address'
                                name='email'
                                autoComplete='email'
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                autoFocus
                            />
                            <TextField
                                margin='normal'
                                required
                                fullWidth
                                name='password'
                                label='Password'
                                type='password'
                                id='password'
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                autoComplete='current-password'
                            />
                            <Button
                                type='submit'
                                fullWidth
                                variant='contained'
                                sx={{ mt: 3, mb: 2 }}>
                                Sign In
                            </Button>
                            <Grid item>
                                <Button
                                    startIcon={<GoogleIcon />}
                                    onClick={() => {
                                        loginWithGoogle();
                                    }}
                                    fullWidth
                                    variant='contained'
                                    sx={{ mb: 2 }}>
                                    Login with Google
                                </Button>
                            </Grid>
                            {error && <p>{error}</p>}
                            <Grid container>
                                <Grid item>
                                    <Link href='/register' variant='body2'>
                                        {"Don't have an account? Sign Up"}
                                    </Link>
                                </Grid>
                            </Grid>
                        </Box>
                    </Box>
                </Container>
                <footer
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "1em",
                        background: "#f1f1f1",
                        width: "100%",
                        marginTop: "auto",
                    }}>
                    <p style={{ margin: 0, textAlign: "center", flex: 1 }}>
                        &copy; {new Date().getFullYear()} Faculdade de Ciências e Tecnologia
                        Universidade NOVA de Lisboa 2024. All rights reserved.
                    </p>

                    <img
                        src='https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png'
                        width='200px'
                        alt='FCT Logo'
                        style={{ marginLeft: "auto" }}
                    />
                </footer>
            </Box>
        </ThemeProvider>
    );
}

export default Login;
