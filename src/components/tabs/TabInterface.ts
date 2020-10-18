import { Vue } from 'vue-property-decorator';

interface TabInterface extends Vue {
  setVisible(value: boolean): void;
}

export default TabInterface;
