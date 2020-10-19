import { Vue } from 'vue-property-decorator';

interface Focusable extends Vue {
  focus(): void;
}

export default Focusable;
