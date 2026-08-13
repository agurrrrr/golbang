use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr;

use golbang_sys::*;

use crate::error::{Error, Result};
use crate::model::Model;

pub type Token = llama_token;

/// Vocab view borrowed from a [`Model`]. Tokenize / detokenize / piece.
#[derive(Clone, Copy)]
pub struct Tokenizer<'a> {
    vocab: *const llama_vocab,
    n_vocab: i32,
    _model: PhantomData<&'a Model>,
}

impl<'a> Tokenizer<'a> {
    pub(crate) fn new(vocab: *const llama_vocab, n_vocab: i32) -> Self {
        Self {
            vocab,
            n_vocab,
            _model: PhantomData,
        }
    }

    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<Token>> {
        let n = unsafe {
            llama_tokenize(
                self.vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        if n == i32::MIN {
            return Err(Error::Tokenize("token count overflow".into()));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let needed = if n < 0 { -n } else { n };
        let mut tokens = vec![0; needed as usize];
        let written = unsafe {
            llama_tokenize(
                self.vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                tokens.as_mut_ptr(),
                needed,
                add_special,
                parse_special,
            )
        };
        if written < 0 {
            return Err(Error::Tokenize(format!(
                "llama_tokenize wrote {written}, probed {needed}"
            )));
        }
        tokens.truncate(written as usize);
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &[Token], remove_special: bool, unparse_special: bool) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; tokens.len().saturating_mul(8).saturating_add(32)];
        loop {
            let n = unsafe {
                llama_detokenize(
                    self.vocab,
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    remove_special,
                    unparse_special,
                )
            };
            if n == i32::MIN {
                return Err(Error::Detokenize("overflow".into()));
            }
            if n < 0 {
                buf.resize((-n) as usize + 8, 0);
                continue;
            }
            buf.truncate(n as usize);
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }
    }

    pub fn token_to_piece(&self, token: Token, special: bool) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 16];
        loop {
            let n = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    special,
                )
            };
            if n < 0 {
                buf.resize((-n) as usize, 0);
                continue;
            }
            buf.truncate(n as usize);
            return Ok(buf);
        }
    }

    pub fn is_eog(&self, token: Token) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab, token) }
    }
}
