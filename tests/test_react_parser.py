"""
Tests for React/TypeScript parser.
"""

import pytest
from pathlib import Path

from src.code_rag.parsers.react_parser import ReactParser
from src.code_rag.parsers.base import EntityType


class TestReactParser:
    """Test suite for ReactParser."""

    @pytest.fixture
    def parser(self):
        """Create React parser instance."""
        return ReactParser()

    def test_parser_supports_react_extensions(self, parser):
        """Test that parser recognizes React file extensions."""
        assert '.jsx' in parser.get_supported_extensions()
        assert '.tsx' in parser.get_supported_extensions()
        assert '.js' in parser.get_supported_extensions()
        assert '.ts' in parser.get_supported_extensions()

    def test_parse_simple_function_component(self, parser, tmp_path):
        """Test parsing a simple React function component."""
        code = '''
import React from 'react';

export const HelloWorld = () => {
    return <div>Hello, World!</div>;
};
'''
        file_path = tmp_path / "HelloWorld.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        assert len(result.entities) >= 1

        component = next(
            (e for e in result.entities if e.type == EntityType.COMPONENT),
            None
        )
        assert component is not None
        assert component.name == "HelloWorld"

    def test_parse_component_with_props(self, parser, tmp_path):
        """Test parsing component with TypeScript props."""
        code = '''
import React from 'react';

interface ButtonProps {
    label: string;
    onClick: () => void;
}

export const Button: React.FC<ButtonProps> = ({ label, onClick }) => {
    return <button onClick={onClick}>{label}</button>;
};
'''
        file_path = tmp_path / "Button.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        component = next(
            (e for e in result.entities if e.name == "Button"),
            None
        )
        assert component is not None
        assert component.metadata['props_type'] == 'ButtonProps'

    def test_parse_component_with_hooks(self, parser, tmp_path):
        """Test parsing component that uses hooks."""
        code = '''
import React, { useState, useEffect } from 'react';

export const Counter = () => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>Increment</button>
        </div>
    );
};
'''
        file_path = tmp_path / "Counter.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        component = next(
            (e for e in result.entities if e.name == "Counter"),
            None
        )
        assert component is not None

        hooks_used = component.metadata['hooks_used']
        hook_names = [h['name'] for h in hooks_used]

        assert 'useState' in hook_names
        assert 'useEffect' in hook_names

    def test_parse_component_with_api_call(self, parser, tmp_path):
        """Test parsing component that makes API calls."""
        code = '''
import React, { useEffect, useState } from 'react';

export const UserList = () => {
    const [users, setUsers] = useState([]);

    useEffect(() => {
        fetch('/api/users')
            .then(res => res.json())
            .then(data => setUsers(data));
    }, []);

    return <div>{users.map(u => <div key={u.id}>{u.name}</div>)}</div>;
};
'''
        file_path = tmp_path / "UserList.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        component = next(
            (e for e in result.entities if e.name == "UserList"),
            None
        )
        assert component is not None

        api_calls = component.metadata['api_calls']
        assert len(api_calls) > 0
        assert any(call['url'] == '/api/users' for call in api_calls)

    def test_parse_component_with_axios(self, parser, tmp_path):
        """Test parsing component using axios."""
        code = '''
import React from 'react';
import axios from 'axios';

export const CreateUser = () => {
    const handleSubmit = async (data) => {
        await axios.post('/api/users', data);
    };

    return <form onSubmit={handleSubmit}>...</form>;
};
'''
        file_path = tmp_path / "CreateUser.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        component = next(
            (e for e in result.entities if e.name == "CreateUser"),
            None
        )
        assert component is not None

        api_calls = component.metadata['api_calls']
        assert len(api_calls) > 0
        assert any(
            call['method'] == 'POST' and call['url'] == '/api/users'
            for call in api_calls
        )

    def test_parse_react_router_routes(self, parser, tmp_path):
        """Test parsing React Router routes."""
        code = '''
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import About from './pages/About';

export const App = () => {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
            </Routes>
        </BrowserRouter>
    );
};
'''
        file_path = tmp_path / "App.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        routes = [e for e in result.entities if e.type == EntityType.ROUTE]
        assert len(routes) >= 2

        route_paths = [r.metadata['path'] for r in routes]
        assert '/' in route_paths
        assert '/about' in route_paths

    def test_parse_component_with_event_handlers(self, parser, tmp_path):
        """Test extracting event handlers."""
        code = '''
import React from 'react';

export const Form = () => {
    const handleSubmit = (e) => {
        e.preventDefault();
    };

    const handleChange = (e) => {
        console.log(e.target.value);
    };

    return (
        <form onSubmit={handleSubmit}>
            <input onChange={handleChange} />
            <button type="submit">Submit</button>
        </form>
    );
};
'''
        file_path = tmp_path / "Form.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        component = next(
            (e for e in result.entities if e.name == "Form"),
            None
        )
        assert component is not None

        handlers = component.metadata['event_handlers']
        assert 'handleSubmit' in handlers
        assert 'handleChange' in handlers

    def test_non_react_file(self, parser, tmp_path):
        """Test that non-React files return empty entities."""
        code = '''
// Just a utility file
export const utils = {
    format: (value) => value.toString()
};
'''
        file_path = tmp_path / "utils.ts"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        # Should parse successfully but find no React components
        assert result.success
        assert len(result.entities) == 0

    def test_multiple_components_in_file(self, parser, tmp_path):
        """Test parsing file with multiple components."""
        code = '''
import React from 'react';

export const Header = () => {
    return <header>Header</header>;
};

export const Footer = () => {
    return <footer>Footer</footer>;
};

export const Layout = ({ children }) => {
    return (
        <div>
            <Header />
            {children}
            <Footer />
        </div>
    );
};
'''
        file_path = tmp_path / "Layout.tsx"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success

        components = [e for e in result.entities if e.type == EntityType.COMPONENT]
        assert len(components) >= 3

        component_names = [c.name for c in components]
        assert 'Header' in component_names
        assert 'Footer' in component_names
        assert 'Layout' in component_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
