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

    // Traverse AST to extract entities
    traverse(ast, {
      // Function declarations and arrow functions
      FunctionDeclaration(path) {
        const node = path.node;
        if (isReactComponent(node, path)) {
          result.components.push(extractComponent(node, path, source));
        } else {
          result.functions.push(extractFunction(node, path, source));
        }
      },

      ArrowFunctionExpression(path) {
        const parent = path.parent;
        if (parent.type === 'VariableDeclarator') {
          const name = parent.id.name;
          if (isReactComponentName(name)) {
            result.components.push(extractComponent(parent, path, source, name));
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
            result.components.push(extractComponent(node, path, source, node.id.name));
          }
        }
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
function extractComponent(node, path, source, explicitName = null) {
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
      const call = extractAPICall(callPath.node, source);
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
function extractAPICall(node, source) {
  const callee = node.callee;

  // fetch('/api/...')
  if (callee.type === 'Identifier' && callee.name === 'fetch') {
    if (node.arguments[0]) {
      const url = extractStringValue(node.arguments[0], source);
      if (url) {
        return { method: 'fetch', url, type: 'fetch' };
      }
    }
  }

  // axios.get('/api/...')
  if (callee.type === 'MemberExpression' &&
      callee.object.name === 'axios') {
    const method = callee.property.name.toUpperCase();
    if (node.arguments[0]) {
      const url = extractStringValue(node.arguments[0], source);
      if (url) {
        return { method, url, type: 'axios' };
      }
    }
  }

  return null;
}

/**
 * Extract string value from AST node (handles template literals)
 */
function extractStringValue(node, source) {
  if (node.type === 'StringLiteral') {
    return node.value;
  }
  if (node.type === 'TemplateLiteral') {
    // Return template as string with ${} placeholders
    return source.substring(node.start, node.end);
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
