#!/usr/bin/env node
/**
 * JavaScript/TypeScript parser using Babel
 *
 * Extracts React components, hooks, API calls, and other entities
 * from JavaScript/TypeScript code using proper AST parsing.
 */

const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const fs = require('fs');

/**
 * Parse JavaScript/TypeScript file and extract entities
 */
function parseFile(filePath, options = {}) {
  const source = fs.readFileSync(filePath, 'utf-8');

  // Babel parser options
  const parserOptions = {
    sourceType: 'module',
    plugins: [
      'jsx',
      'typescript',
      'classProperties',
      'decorators-legacy',
      'dynamicImport',
      'optionalChaining',
      'nullishCoalescingOperator',
    ],
  };

  try {
    const ast = parser.parse(source, parserOptions);

    const result = {
      imports: [],
      components: [],
      functions: [],
      variables: [],
      routes: [],
      errors: [],
    };

    // Extract imports
    ast.program.body.forEach(node => {
      if (node.type === 'ImportDeclaration') {
        result.imports.push({
          source: node.source.value,
          specifiers: node.specifiers.map(spec => ({
            local: spec.local.name,
            imported: spec.imported ? spec.imported.name : null,
            type: spec.type, // ImportDefaultSpecifier, ImportSpecifier, ImportNamespaceSpecifier
          })),
        });
      }
    });

    // Сбор URL-констант на верхнем уровне модуля.
    // Паттерн: const api_url_X = ConfigVariables.API_URL + '/backend/foo/'
    // → urlVars['api_url_X'] = '/backend/foo/'
    // Игнорируем не-литеральные части (например, идентификаторы host'а) —
    // склеиваем только StringLiteral'ы из BinaryExpression и TemplateLiteral.
    const urlVars = {};
    ast.program.body.forEach(node => {
      if (node.type !== 'VariableDeclaration') return;
      for (const decl of node.declarations) {
        if (!decl.id || decl.id.type !== 'Identifier') continue;
        if (!decl.init) continue;
        const url = collectStringLiterals(decl.init);
        if (url) urlVars[decl.id.name] = url;
      }
    });

    // Traverse AST to extract entities
    traverse(ast, {
      // Function declarations and arrow functions
      FunctionDeclaration(path) {
        const node = path.node;
        if (isReactComponent(node, path)) {
          result.components.push(extractComponent(node, path, source, null, urlVars));
        } else {
          result.functions.push(extractFunction(node, path, source));
        }
      },

      ArrowFunctionExpression(path) {
        const parent = path.parent;
        if (parent.type === 'VariableDeclarator') {
          const name = parent.id.name;
          if (isReactComponentName(name)) {
            result.components.push(extractComponent(parent, path, source, name, urlVars));
          }
        }
      },

      // Variable declarations (for const Component = () => {})
      VariableDeclarator(path) {
        const node = path.node;
        if (node.id.name && isReactComponentName(node.id.name)) {
          if (node.init && (
            node.init.type === 'ArrowFunctionExpression' ||
            node.init.type === 'FunctionExpression'
          )) {
            result.components.push(extractComponent(node, path, source, node.id.name, urlVars));
          }
        }
      },

      // Class components: class Foo extends Component / React.Component / PureComponent
      ClassDeclaration(path) {
        const node = path.node;
        if (!node.id || !isReactComponentName(node.id.name)) return;
        if (!isReactClassComponent(node)) return;
        result.components.push(extractComponent(node, path, source, node.id.name, urlVars));
      },

      // JSX elements for routes
      JSXElement(path) {
        const openingElement = path.node.openingElement;
        if (openingElement.name.name === 'Route') {
          result.routes.push(extractRoute(path.node, source));
        }
      },
    });

    return result;

  } catch (error) {
    return {
      imports: [],
      components: [],
      functions: [],
      variables: [],
      routes: [],
      errors: [`Parse error: ${error.message}`],
    };
  }
}

/**
 * Check if function is a React component
 */
function isReactComponent(node, path) {
  // Check name (starts with capital letter)
  if (node.id && isReactComponentName(node.id.name)) {
    // Check if it returns JSX
    return containsJSX(path);
  }
  return false;
}

/**
 * Check if name follows React component naming (PascalCase)
 */
function isReactComponentName(name) {
  return name && /^[A-Z]/.test(name);
}

/**
 * Check if a ClassDeclaration is a React class component.
 *
 * Признаки:
 *   class X extends Component
 *   class X extends React.Component
 *   class X extends PureComponent
 *   class X extends React.PureComponent
 */
function isReactClassComponent(node) {
  const sup = node.superClass;
  if (!sup) return false;
  if (sup.type === 'Identifier') {
    return sup.name === 'Component' || sup.name === 'PureComponent';
  }
  if (sup.type === 'MemberExpression'
      && sup.object && sup.object.name === 'React'
      && sup.property && sup.property.type === 'Identifier') {
    return sup.property.name === 'Component' || sup.property.name === 'PureComponent';
  }
  return false;
}

/**
 * Check if path contains JSX
 */
function containsJSX(path) {
  let hasJSX = false;
  path.traverse({
    JSXElement() {
      hasJSX = true;
    },
    JSXFragment() {
      hasJSX = true;
    },
  });
  return hasJSX;
}

/**
 * Extract component information
 */
function extractComponent(node, path, source, explicitName = null, urlVars = {}) {
  const name = explicitName || (node.id ? node.id.name : 'AnonymousComponent');

  // Get location
  const loc = node.loc || (node.init && node.init.loc);
  const start = loc ? loc.start : { line: 1, column: 0 };
  const end = loc ? loc.end : { line: 1, column: 0 };

  // Extract code
  const code = source.substring(
    getPositionInSource(source, start.line, start.column),
    getPositionInSource(source, end.line, end.column)
  );

  // Extract props type
  let propsType = null;
  const funcNode = node.init || node;
  if (funcNode.params && funcNode.params[0]) {
    const param = funcNode.params[0];
    if (param.typeAnnotation) {
      propsType = source.substring(
        param.typeAnnotation.start,
        param.typeAnnotation.end
      );
    }
  }

  // Extract hooks
  const hooks = [];
  path.traverse({
    CallExpression(hookPath) {
      const callee = hookPath.node.callee;
      if (callee.type === 'Identifier' && callee.name.startsWith('use')) {
        hooks.push({
          name: callee.name,
          line: hookPath.node.loc.start.line,
        });
      }
    },
  });

  // Extract API calls
  const apiCalls = [];
  path.traverse({
    CallExpression(callPath) {
      const call = extractAPICall(callPath.node, source, urlVars);
      if (call) {
        apiCalls.push(call);
      }
    },
  });

  // Extract event handlers
  const handlers = new Set();
  path.traverse({
    JSXAttribute(attrPath) {
      const attr = attrPath.node;
      if (attr.name && attr.name.name && attr.name.name.startsWith('on')) {
        if (attr.value && attr.value.expression && attr.value.expression.name) {
          handlers.add(attr.value.expression.name);
        }
      }
    },
  });

  return {
    type: 'component',
    name,
    start_line: start.line,
    end_line: end.line,
    code,
    props_type: propsType,
    hooks: hooks,
    api_calls: apiCalls,
    event_handlers: Array.from(handlers),
    is_exported: isExported(path),
  };
}

/**
 * Extract function information
 */
function extractFunction(node, path, source) {
  const name = node.id ? node.id.name : 'anonymous';
  const start = node.loc.start;
  const end = node.loc.end;

  const code = source.substring(
    getPositionInSource(source, start.line, start.column),
    getPositionInSource(source, end.line, end.column)
  );

  return {
    type: 'function',
    name,
    start_line: start.line,
    end_line: end.line,
    code,
    is_async: node.async,
    is_exported: isExported(path),
  };
}

/**
 * Extract API call from CallExpression
 */
function extractAPICall(node, source, urlVars = {}) {
  const callee = node.callee;
  const HTTP_METHODS = new Set(['get', 'post', 'put', 'patch', 'delete', 'head', 'options']);

  // fetch('/api/...', { method: 'POST' })
  if (callee.type === 'Identifier' && callee.name === 'fetch') {
    if (node.arguments[0]) {
      const url = resolveURLArgument(node.arguments[0], source, urlVars);
      if (url) {
        // Default to GET; lift `method` from options object if present.
        let method = 'GET';
        const opts = node.arguments[1];
        if (opts && opts.type === 'ObjectExpression') {
          for (const prop of opts.properties) {
            const keyName =
              (prop.key && (prop.key.name || prop.key.value)) || null;
            if (keyName === 'method' && prop.value) {
              if (prop.value.type === 'StringLiteral') {
                method = prop.value.value.toUpperCase();
              } else if (prop.value.type === 'TemplateLiteral'
                  && prop.value.quasis.length === 1) {
                method = prop.value.quasis[0].value.cooked.toUpperCase();
              }
              break;
            }
          }
        }
        return { method, url, type: 'fetch' };
      }
    }
  }

  // <Service>.<method>('/api/...' | api_url_X | template) — самый частый паттерн
  // (axios.get / api.post / UploadDataService.put / GetDataService.get / ...).
  // service-имя берём из callee.object.name; если object — это MemberExpression
  // (например, this.api.post(...)), берём property.name последнего звена.
  if (callee.type === 'MemberExpression'
      && callee.property && callee.property.type === 'Identifier'
      && HTTP_METHODS.has(callee.property.name)) {
    const method = callee.property.name.toUpperCase();
    if (node.arguments[0]) {
      const url = resolveURLArgument(node.arguments[0], source, urlVars);
      if (url) {
        const obj = callee.object;
        let serviceName = 'http';
        if (obj.type === 'Identifier') serviceName = obj.name;
        else if (obj.type === 'MemberExpression' && obj.property && obj.property.name) {
          serviceName = obj.property.name;
        } else if (obj.type === 'ThisExpression') {
          serviceName = 'this';
        }
        return { method, url, type: serviceName };
      }
    }
  }

  // axios({ url: '/api/...', method: 'POST' })
  if (callee.type === 'Identifier' && callee.name === 'axios'
      && node.arguments[0] && node.arguments[0].type === 'ObjectExpression') {
    let url = null;
    let method = 'GET';
    for (const prop of node.arguments[0].properties) {
      const keyName =
        (prop.key && (prop.key.name || prop.key.value)) || null;
      if (keyName === 'url' && prop.value) {
        const u = resolveURLArgument(prop.value, source, urlVars);
        if (u) url = u;
      } else if (keyName === 'method' && prop.value
          && prop.value.type === 'StringLiteral') {
        method = prop.value.value.toUpperCase();
      }
    }
    if (url) {
      return { method, url, type: 'axios' };
    }
  }

  return null;
}

/**
 * Resolve URL argument: literal, template, или identifier из ранее собранной
 * map URL-констант модуля.
 *
 *   'literal'           → 'literal'
 *   `${X}/literal`      → '${X}/literal' (template-as-source) или {param} после нормализации
 *   api_url_foo         → urlVars['api_url_foo'] (если есть)
 *   API_BASE + '/path'  → '/path' (через collectStringLiterals)
 */
function resolveURLArgument(node, source, urlVars) {
  if (!node) return null;
  let url = null;
  if (node.type === 'StringLiteral') url = node.value;
  else if (node.type === 'TemplateLiteral') url = source.substring(node.start, node.end);
  else if (node.type === 'Identifier') url = urlVars[node.name] || null;
  else if (node.type === 'BinaryExpression' && node.operator === '+') {
    url = collectStringLiterals(node);
  }
  // Отсекаем явно не-API URL'ы:
  //  - пустые / тривиальный '/'.
  //  - без слеша ('dataAdapter', 'vanilla', 'exotic' — react-select
  //    adapter-имена, не пути).
  //  - голые template-literal'ы вида `${url}${id}/` — без литерального
  //    префикса пути это runtime-составленные URL'ы, матчить нечем.
  if (!url) return null;
  const trimmed = url.trim();
  if (trimmed === '' || trimmed === '/') return null;
  if (!trimmed.includes('/')) return null;
  // Валидный путь начинается с '/', './', '../' или схемы (http, https, //).
  // Backtick-обёрнутый template (`${...}`) — отбрасываем.
  const head = trimmed.replace(/^[`'"]/, '');
  if (!/^(\/|\.\/|\.\.\/|https?:\/\/|\/\/)/.test(head)) return null;
  return url;
}

/**
 * Backward-compat: используется в extractRoute и других местах.
 */
function extractStringValue(node, source) {
  if (node.type === 'StringLiteral') return node.value;
  if (node.type === 'TemplateLiteral') return source.substring(node.start, node.end);
  return null;
}

/**
 * Recursively collect StringLiteral values from BinaryExpression/TemplateLiteral.
 * Игнорируем все Identifier'ы и MemberExpression'ы (host, baseURL и т.п.).
 */
function collectStringLiterals(node) {
  if (!node) return null;
  if (node.type === 'StringLiteral') return node.value;
  if (node.type === 'TemplateLiteral') {
    // склеиваем cooked-куски (всё что вне ${...})
    return node.quasis.map(q => q.value.cooked).join('');
  }
  if (node.type === 'BinaryExpression' && node.operator === '+') {
    const left = collectStringLiterals(node.left) || '';
    const right = collectStringLiterals(node.right) || '';
    const combined = left + right;
    return combined || null;
  }
  return null;
}

/**
 * Extract route from Route JSX element
 */
function extractRoute(node, source) {
  const attrs = node.openingElement.attributes;
  let path = null;
  let component = null;

  attrs.forEach(attr => {
    if (attr.type === 'JSXAttribute') {
      if (attr.name.name === 'path' && attr.value) {
        if (attr.value.type === 'StringLiteral') {
          path = attr.value.value;
        }
      }
      if (attr.name.name === 'element' && attr.value) {
        if (attr.value.type === 'JSXExpressionContainer') {
          const expr = attr.value.expression;
          if (expr.type === 'JSXElement') {
            component = expr.openingElement.name.name;
          }
        }
      }
    }
  });

  return {
    type: 'route',
    path,
    component,
    line: node.loc.start.line,
  };
}

/**
 * Check if node is exported
 */
function isExported(path) {
  let current = path;
  while (current) {
    if (current.parent && current.parent.type === 'ExportNamedDeclaration') {
      return true;
    }
    if (current.parent && current.parent.type === 'ExportDefaultDeclaration') {
      return true;
    }
    current = current.parentPath;
  }
  return false;
}

/**
 * Get character position in source from line and column
 */
function getPositionInSource(source, line, column) {
  const lines = source.split('\n');
  let pos = 0;
  for (let i = 0; i < line - 1 && i < lines.length; i++) {
    pos += lines[i].length + 1; // +1 for newline
  }
  pos += column;
  return pos;
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('Usage: node js_parser.js <file_path>');
    process.exit(1);
  }

  const filePath = args[0];

  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
  }

  const result = parseFile(filePath);
  console.log(JSON.stringify(result, null, 2));
}

module.exports = { parseFile };
