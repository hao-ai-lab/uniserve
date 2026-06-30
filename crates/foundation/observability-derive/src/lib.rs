use proc_macro::TokenStream;
use quote::quote;
use syn::Data;
use syn::DeriveInput;
use syn::Error;
use syn::Expr;
use syn::Fields;
use syn::Ident;
use syn::LitStr;
use syn::Token;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;

/// Derive a `register` constructor that registers every field into a
/// `prometheus_client::registry::Registry`.
///
/// Each field carries `#[metric(name = "...", help = "...")]`, where `name` is
/// the Prometheus metric name and `help` is the metric help text. Both are
/// passed verbatim to `Registry::register`. An optional `init = <expr>`
/// supplies the field's initial value; without it the field is built with
/// `Default::default()`.
///
/// The generated method is:
///
/// ```ignore
/// pub(crate) fn register(registry: &mut ::prometheus_client::registry::Registry) -> Self
/// ```
///
/// It constructs every field, registers a clone with the supplied name and
/// help, and returns `Self`.
#[proc_macro_derive(MetricFamily, attributes(metric))]
pub fn derive_metric_family(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

struct MetricField {
    ident: Ident,
    name: LitStr,
    help: LitStr,
    init: Option<Expr>,
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(Error::new(
                    input.span(),
                    "MetricFamily can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(Error::new(
                input.span(),
                "MetricFamily can only be derived for structs",
            ));
        }
    };

    let mut metric_fields = Vec::with_capacity(fields.len());
    for field in fields {
        metric_fields.push(parse_field(field)?);
    }

    let registrations = metric_fields.iter().map(|field| {
        let ident = &field.ident;
        let name = &field.name;
        let help = &field.help;
        let init = match &field.init {
            Some(expr) => quote!(#expr),
            None => quote!(::core::default::Default::default()),
        };
        quote! {
            let #ident = #init;
            registry.register(#name, #help, ::core::clone::Clone::clone(&#ident));
        }
    });

    let field_idents = metric_fields.iter().map(|field| &field.ident);

    Ok(quote! {
        impl #impl_generics #struct_ident #ty_generics #where_clause {
            pub(crate) fn register(registry: &mut ::prometheus_client::registry::Registry) -> Self {
                #(#registrations)*
                Self {
                    #(#field_idents),*
                }
            }
        }
    })
}

fn parse_field(field: &syn::Field) -> syn::Result<MetricField> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| Error::new(field.span(), "MetricFamily fields must be named"))?;

    let mut name: Option<LitStr> = None;
    let mut help: Option<LitStr> = None;
    let mut init: Option<Expr> = None;

    let mut metric_attr_seen = false;
    for attr in &field.attrs {
        if !attr.path().is_ident("metric") {
            continue;
        }
        metric_attr_seen = true;
        let metas = attr.parse_args_with(Punctuated::<MetricArg, Token![,]>::parse_terminated)?;
        for meta in metas {
            match meta {
                MetricArg::Name(value) => {
                    if name.is_some() {
                        return Err(Error::new(value.span(), "duplicate `name` in #[metric]"));
                    }
                    name = Some(value);
                }
                MetricArg::Help(value) => {
                    if help.is_some() {
                        return Err(Error::new(value.span(), "duplicate `help` in #[metric]"));
                    }
                    help = Some(value);
                }
                MetricArg::Init(value) => {
                    if init.is_some() {
                        return Err(Error::new(value.span(), "duplicate `init` in #[metric]"));
                    }
                    init = Some(value);
                }
            }
        }
    }

    if !metric_attr_seen {
        return Err(Error::new(
            field.span(),
            "every MetricFamily field needs a #[metric(name = \"...\", help = \"...\")] attribute",
        ));
    }

    let name = name.ok_or_else(|| Error::new(ident.span(), "#[metric] is missing `name`"))?;
    let help = help.ok_or_else(|| Error::new(ident.span(), "#[metric] is missing `help`"))?;

    Ok(MetricField {
        ident,
        name,
        help,
        init,
    })
}

enum MetricArg {
    Name(LitStr),
    Help(LitStr),
    Init(Expr),
}

impl syn::parse::Parse for MetricArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        if key == "name" {
            Ok(MetricArg::Name(input.parse()?))
        } else if key == "help" {
            Ok(MetricArg::Help(input.parse()?))
        } else if key == "init" {
            Ok(MetricArg::Init(input.parse()?))
        } else {
            Err(Error::new(
                key.span(),
                "unknown #[metric] key (expected `name`, `help`, or `init`)",
            ))
        }
    }
}
